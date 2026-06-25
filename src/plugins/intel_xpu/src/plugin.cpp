// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xpu_debug.hpp"
#include "plugin.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <regex>

#include "compiled_model.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_weight_buffer.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
//
#    include <psapi.h>
#endif

namespace ov::intel_xpu {

namespace {
// Process private/committed bytes (MB) — for the [XPU][MEM] compile-step breakdown.
double xpu_committed_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
        return static_cast<double>(pmc.PrivateUsage) / 1048576.0;
#endif
    return 0.0;
}
#define XPU_MEM(tag)                                                                                  \
    do {                                                                                              \
        if (std::getenv("XPU_MEM_DEBUG"))                                                             \
            std::cout << "[XPU][MEM] " << (tag) << ": " << xpu_committed_mb() << " MB committed"      \
                      << std::endl;                                                                   \
    } while (0)

// --- GPU KV-write (always on) -------------------------------------------------------------------
// Rewrite the GPU prefill model's present.* outputs so the GPU itself produces the NPU's static KV
// slot (f16, value-transposed) and can write it zero-copy into the shared KV buffer, replacing the
// CPU cast/transpose/scatter bridge. Per present.N.{key,value}: Convert f32->f16, Transpose value
// to [1,h,d,seq], then ScatterUpdate the dynamic-seq K/V into rows [0,seq) of a Broadcast(0) static
// slot. The static output shape is what lets it bind zero-copy (intel_gpu prepare_output).
struct XpuKvSlot {
    std::string name;  // past_key_values.N.{key,value}
    ov::Shape shape;   // static slot shape, e.g. [1,heads,total_size,head_dim]
};

std::vector<XpuKvSlot> xpu_parse_kv_layout(const std::string& layout) {
    std::vector<XpuKvSlot> slots;
    std::istringstream ss(layout);
    std::string entry;
    while (std::getline(ss, entry, ';')) {
        if (entry.empty())
            continue;
        std::istringstream es(entry);
        std::string name, offset_s, type_s, shape_s;
        std::getline(es, name, ':');
        std::getline(es, offset_s, ':');
        std::getline(es, type_s, ':');
        std::getline(es, shape_s, ':');
        XpuKvSlot slot;
        slot.name = name;
        std::istringstream sh(shape_s);
        std::string dim;
        while (std::getline(sh, dim, ','))
            slot.shape.push_back(static_cast<size_t>(std::stoull(dim)));
        slots.push_back(std::move(slot));
    }
    return slots;
}

void xpu_rewrite_present_to_static_slots(const std::shared_ptr<ov::Model>& m,
                                         const std::vector<XpuKvSlot>& slots,
                                         bool v_transposed) {
    using namespace ov::op;
    auto find_slot = [&](const std::string& n) -> const XpuKvSlot* {
        for (const auto& s : slots)
            if (s.name == n)
                return &s;
        return nullptr;
    };
    size_t rewritten = 0;
    for (const auto& result : m->get_results()) {
        std::string pres;
        for (const auto& nm : result->get_output_tensor(0).get_names()) {
            if (nm.find("present") != std::string::npos) {
                pres = nm;
                break;
            }
        }
        if (pres.empty())
            continue;
        const std::string pkv = std::regex_replace(pres, std::regex("present"), "past_key_values");
        const XpuKvSlot* slot = find_slot(pkv);
        if (!slot || slot->shape.size() != 4)
            continue;

        const bool is_value = pres.find("value") != std::string::npos;
        const bool transpose = is_value && v_transposed;
        const int64_t axis = transpose ? 3 : 2;

        ov::Output<ov::Node> src = result->input_value(0);  // [1,heads,seq,head_dim] f32 (dynamic seq)
        std::shared_ptr<ov::Node> upd = std::make_shared<v0::Convert>(src, ov::element::f16);
        if (transpose) {
            auto order = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
            upd = std::make_shared<v1::Transpose>(upd, order);
        }
        // seq (dynamic) = ShapeOf(upd)[axis]
        auto shp = std::make_shared<v3::ShapeOf>(upd, ov::element::i64);
        auto axis_idx = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        auto g0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        std::shared_ptr<ov::Node> seq = std::make_shared<v8::Gather>(shp, axis_idx, g0);  // scalar
        auto c0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto c1 = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto indices = std::make_shared<v4::Range>(c0, seq, c1, ov::element::i64);  // [seq]
        // static zeros of the slot shape (Broadcast a scalar 0 -> no baked constant data)
        std::vector<int64_t> slot_dims(slot->shape.begin(), slot->shape.end());
        auto tgt = v0::Constant::create(ov::element::i64, ov::Shape{4}, slot_dims);
        auto zero = v0::Constant::create(ov::element::f16, ov::Shape{}, {0});
        auto data = std::make_shared<v3::Broadcast>(zero, tgt);
        auto su_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        auto scatter = std::make_shared<v3::ScatterUpdate>(data, indices, upd, su_axis);
        scatter->set_friendly_name(pres + "_xpu_kvslot");
        result->set_argument(0, scatter->output(0));
        ++rewritten;
    }
    m->validate_nodes_and_infer_types();
    xpu_dbg() << "[XPU] step 6f: rewrote " << rewritten
              << " present.* outputs to static NPUW KV slots (GPU KV-write)" << std::endl;
}

// Slice the prefill logits to the LAST token (port of OpenVINO GenAI's apply_slice_before_matmul,
// utils.cpp). The prefill only needs logits[-1] (the first generated token = argmax(logits[-1])); by
// slicing the hidden-state input of the LM-head MatMul to the last sequence position, the LM-head matmul
// and the logits output both shrink from [1, seq, vocab] to [1, 1, vocab] -- removing the wasted
// full-sequence LM head (part of the GPU compute) AND the ~622 MB logits materialization at 1K.
// Returns true if the slice was applied.
bool xpu_slice_prefill_logits_to_last(const std::shared_ptr<ov::Model>& m) {
    // Locate the Result producing "logits" (the model also has present.* KV Results, so don't assume
    // output(0)).
    ov::Output<ov::Node> logits_in;
    bool found = false;
    for (const auto& res : m->get_results()) {
        const auto src = res->input_value(0);
        const auto& res_names = res->get_output_tensor(0).get_names();
        if (res_names.count("logits") > 0 || src.get_names().count("logits") > 0) {
            logits_in = src;
            found = true;
            break;
        }
    }
    if (!found)
        return false;

    // Walk back through the known LM-head tail patterns to the MatMul: MatMul->Result,
    // MatMul->Add->Result (bias), MatMul->Transpose->Result.
    auto tail = logits_in.get_node_shared_ptr();
    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(tail);
    int64_t seq_dim = 1;  // [batch, seq, hidden]
    if (!matmul) {
        if (auto add = ov::as_type_ptr<ov::op::v1::Add>(tail)) {
            matmul = ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
        } else if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(tail)) {
            matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
            if (auto order =
                    ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr())) {
                const auto perm = order->get_axis_vector_val();
                if (seq_dim < static_cast<int64_t>(perm.size()))
                    seq_dim = static_cast<int64_t>(perm[seq_dim]);
            }
        }
    }
    if (!matmul)
        return false;
    // Only slice when the activation input is rank-3 [batch, seq, hidden] (no-op / unsafe otherwise).
    const auto act_rank = matmul->input(0).get_partial_shape().rank();
    if (act_rank.is_dynamic() || act_rank.get_length() != 3)
        return false;

    // Slice(start=-1, stop=-2, step=-1, axis=seq_dim): keep exactly the last sequence position.
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {int64_t(-1)});
    auto stop = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {int64_t(-2)});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {int64_t(-1)});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {seq_dim});
    auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
    slice->set_friendly_name(matmul->get_friendly_name() + "_xpu_last_token");
    matmul->input(0).replace_source_output(slice->output(0));
    m->validate_nodes_and_infer_types();
    return true;
}
}  // namespace

Plugin::Plugin() {
    set_device_name("XPU");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::filesystem::path& model_path,
                                                           const ov::AnyMap& properties) const {
    // XPU compile from file path: the primary entry point.
    xpu_dbg() << "[XPU] compile_model: start, path=" << model_path << std::endl;

    // 1. Derive .bin path from .xml path
    auto xml_path = model_path;
    auto bin_path = std::filesystem::path(model_path).replace_extension(".bin");
    OPENVINO_ASSERT(std::filesystem::exists(xml_path), "[XPU] Model file not found: ", xml_path.string());
    OPENVINO_ASSERT(std::filesystem::exists(bin_path), "[XPU] Weights file not found: ", bin_path.string());
    xpu_dbg() << "[XPU] step 1: paths ok, xml=" << xml_path << " bin=" << bin_path << std::endl;

    auto core = get_core();
    OPENVINO_ASSERT(core, "[XPU] ICore is not set");
    ov::SoPtr<ov::IRemoteContext> gpu_ctx;
    try {
        gpu_ctx = core->get_default_context("GPU");
    } catch (const std::exception& e) {
        xpu_dbg() << "[XPU] step 2: GPU context unavailable (" << e.what()
                  << "), using fallback allocation" << std::endl;
    }

    // 2. Read the model with the .bin memory-mapped (lazy paging — the whole file is never
    //    resident). The constants view the mmap until we relocate them below.
    auto model = core->read_model(xml_path, bin_path, {});
    xpu_dbg() << "[XPU] step 2: model read (mmap), " << model->get_ops().size() << " ops" << std::endl;
    XPU_MEM("after read_model");

    // Upgrade IR version to 11 if < 11 (required by NPU plugin).
    if (model->has_rt_info("version")) {
        auto ir_version = model->get_rt_info<int64_t>("version");
        if (ir_version < 11) {
            model->set_rt_info(int64_t(11), "version");
            xpu_dbg() << "[XPU] step 2: upgraded IR version " << ir_version << " -> 11" << std::endl;
        }
    }

    // 3. Relocate each large weight Constant into its OWN GPU-USM-host buffer.
    //    Why per-weight buffers: the GPU cannot reliably address a single host-USM allocation
    //    larger than ~2GB (a 2^31 boundary in the OCL/cldnn USM path; the device max-alloc is
    //    far larger, so this is a software limit, not capacity). Each weight is < 2GB, so a
    //    per-weight buffer is always GPU-addressable. We copy each constant from the mmap into
    //    its buffer, rebind the Constant to VIEW that buffer (zero-copy, non-copying ctor), and
    //    move on — the mmap page is then reclaimed, so peak footprint ~= total weights (not 2x).
    //    Both engines then read these buffers: GPU via share_usm, NPU via host-unpack/raw-share.
    std::vector<std::shared_ptr<SharedWeightBuffer>> weight_segments;
    std::string weight_ranges;  // serialized "ptr:size;ptr:size;..."
    {
        constexpr size_t kMinRelocateBytes = 4096;  // page-sized+; skip tiny scalar constants
        std::vector<std::shared_ptr<ov::op::v0::Constant>> consts;
        for (const auto& op : model->get_ops()) {
            auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
            if (c && c->get_byte_size() >= kMinRelocateBytes)
                consts.push_back(c);
        }
        std::ostringstream rs;
        size_t total_bytes = 0;
        for (auto& c : consts) {
            const size_t bytes = c->get_byte_size();
            auto buf = SharedWeightBuffer::create_empty(gpu_ctx, bytes, /*zero=*/false);
            std::memcpy(buf->host_ptr, c->get_data_ptr(), bytes);
            auto so = std::static_pointer_cast<void>(buf);  // keeps the USM buffer alive
            auto new_c = std::make_shared<ov::op::v0::Constant>(
                c->get_element_type(), c->get_shape(), buf->host_ptr, so);
            new_c->set_friendly_name(c->get_friendly_name());
            ov::copy_runtime_info(c, new_c);
            // WeightlessCacheAttribute::is_copyable()==false, so copy_runtime_info drops it. Without it,
            // NPUW's Const wrapper (lazy_tensor.cpp) treats every relocated weight as a "new Constant not
            // in the weights file" and eagerly copies it to host (~2.1 GB second copy at partition time).
            // The relocated buffer is a verbatim memcpy of .bin[bin_offset], so preserving the attr is
            // correct (and keeps weightless caching valid).
            ov::copy_weightless_cache_attr(c, new_c);
            ov::replace_node(c, new_c);
            if (!rs.str().empty()) rs << ";";
            rs << reinterpret_cast<uint64_t>(buf->host_ptr) << ":" << buf->alloc_size;
            weight_segments.push_back(std::move(buf));
            total_bytes += bytes;
        }
        weight_ranges = rs.str();
        xpu_dbg() << "[XPU] step 3: relocated " << weight_segments.size()
                  << " weight constants into per-weight USM buffers ("
                  << (total_bytes / 1048576.0) << " MB)" << std::endl;
    }
    XPU_MEM("after relocate weights");

    // For the GPU to read the shared i4 weights with NO reorder/copy (FullyConnected_bfyx_Ref instead
    // of bf_tiled's blocked reorder), the kernel selector reads OV_XPU_REF_COMPRESSED_FC via getenv at
    // GPU compile time. Native INT4 is the default path, so set it here unless the app overrode it.
#ifdef _WIN32
    if (std::getenv("OV_XPU_REF_COMPRESSED_FC") == nullptr) {
        _putenv_s("OV_XPU_REF_COMPRESSED_FC", "1");
    }
#endif

    // 5. rt_info: the set of shared weight buffer ranges (GPU share_usm membership test).
    //    DEFAULT = page-aligned malloc imported into BOTH the GPU-L0 and NPU-L0 contexts (one physical
    //    buffer, zero-copy native on both engines). Opt out to legacy OCL USM-host with XPU_USM_WEIGHTS=1
    //    (GPU OCL share_usm + NPU host-unpack only — used by the DCOFF concept-proof path; see appendix).
    //    With malloc, GPU can share_usm ONLY if every buffer was imported into the GPU-L0 context; if not
    //    (e.g. OCL GPU build), GPU copies while the NPU still imports the malloc via XPU_RAW_WEIGHT_RANGES.
    const bool malloc_weights = (std::getenv("XPU_USM_WEIGHTS") == nullptr);
    bool gpu_shares = !malloc_weights;
    if (malloc_weights && !weight_segments.empty()) {
        gpu_shares = true;
        for (const auto& s : weight_segments) {
            if (!s->gpu_l0_imported) {
                gpu_shares = false;
                break;
            }
        }
        xpu_dbg() << "[XPU] step 5: malloc weights imported into GPU-L0 context: "
                  << (gpu_shares ? "YES -> GPU + NPU share ONE physical buffer (zero-copy both)"
                                 : "NO -> GPU copies, NPU shares")
                  << std::endl;
    }
    if (gpu_shares) {
        model->set_rt_info(weight_ranges, "xpu_shared_weight_ranges");
    }

    // 6. Compile NPU first (stricter format constraints)
    //    Forward application-provided properties (e.g. NPUW configs for LLMs)
    //    and layer on XPU-required defaults (compiler type, NPUW).
    xpu_dbg() << "[XPU] step 6: compiling NPU..." << std::endl;
    ov::AnyMap npu_props;
    for (const auto& [key, val] : properties) {
        // Forward NPU/NPUW properties from application; skip XPU-only keys
        if (key.find("NPU") != std::string::npos || key.find("NPUW") != std::string::npos ||
            key.find("LOG_LEVEL") != std::string::npos ||
            key.find("MAX_PROMPT_LEN") != std::string::npos ||
            key.find("MIN_RESPONSE_LEN") != std::string::npos) {
            npu_props[key] = val;
        }
    }
    // NPU_USE_NPUW activates the NPUW path in the NPU plugin.
    // Without this, the NPU plugin uses its native compilation path which
    // validates model shapes strictly (fails on LLM dynamic KV-cache shapes).
    npu_props["NPU_USE_NPUW"] = true;
    npu_props["NPUW_LLM"] = true;
    // Weight execution path. DEFAULT = NATIVE INT4: leave DCOFF off so NPUW's baseline runs the
    // compiler-dynamic-quant path (NPUW_DQ=YES + NPU_COMPILER_DYNAMIC_QUANTIZATION=YES) — weights stay
    // i4 and the VCL compiler emits an on-device INT4 matmul (no host u4->f16 unpack, no 4x f16
    // expansion). Requires an i4-SYMMETRIC, per-channel (group-size -1) export.
    //
    // OPT-IN DCOFF (XPU_DCOFF=1): force DCOFF (host u4->f16 unpack each inference). This is the LEGACY
    // concept-proof path — it keeps weights u4 in the shared buffer but executes f16 on the NPU
    // (slower, 4x device expansion) and needs a u4-ASYMMETRIC export. Kept only as a working fallback /
    // KV-cache demonstration; see the DCOFF appendix in the architecture report.
    if (std::getenv("XPU_DCOFF") != nullptr) {
        npu_props["NPUW_DCOFF_TYPE"] = "f16";
        npu_props["NPUW_DCOFF_SCALE"] = "YES";
        xpu_dbg() << "[XPU] weight path: DCOFF host-unpack to f16 (legacy fallback) [XPU_DCOFF=1]" << std::endl;
    } else {
        xpu_dbg() << "[XPU] weight path: NATIVE INT4 (compiler dynamic-quant, on-device) [default]" << std::endl;
    }
    // NPUW loads full weights during compilation. Consolidation into shared
    // buffer happens via set_property() after compilation completes.

    // NPUW_DEVICES: which device executes the partitioned sub-models.
    // History: VCL 7.7.0 miscompiled this INT4 LLM (compiled OK but produced wrong
    // logits, e.g. repeating token 22931), which forced a temporary NPUW_DEVICES=CPU
    // fallback. Upgrading the bundled NPU plugin compiler to VCL 8.1.0 (see
    // intel_npu/cmake/download_compiler_libs.cmake) FIXES the miscompile: the NPU now
    // produces output bit-identical to the CPU reference. We therefore run sub-models
    // on the real NPU. An application may override by passing NPUW_DEVICES explicitly.
    if (npu_props.find("NPUW_DEVICES") == npu_props.end()) {
        npu_props["NPUW_DEVICES"] = "NPU";
    }
    // Force the in-process PLUGIN compiler so the upgraded VCL 8.1.0 binary is used
    // regardless of platform default. On NPU3720 the default is the DRIVER-embedded
    // VCL (still the older, miscompiling build); PLUGIN guarantees the fixed compiler.
    if (npu_props.find("NPU_COMPILER_TYPE") == npu_props.end()) {
        npu_props["NPU_COMPILER_TYPE"] = "PLUGIN";
    }

    // Hybrid GPU-prefill: skip compiling the NPU prefill model BY DEFAULT (the GPU performs prefill, so
    // the NPU prefill model is never executed — it only wastes a ~2GB compile). Keep it ONLY for the
    // pure-NPUW path (XPU_ACTIVE_DEVICE=NPU at compile, which DOES run NPU prefill) or XPU_KEEP_NPU_PREFILL=1.
    const bool active_npu = (properties.count("XPU_ACTIVE_DEVICE") &&
                             properties.at("XPU_ACTIVE_DEVICE").as<std::string>() == "NPU");
    if (!active_npu && std::getenv("XPU_KEEP_NPU_PREFILL") == nullptr) {
        npu_props["XPU_SKIP_NPU_PREFILL"] = true;
    }

    // Pass the shared-buffer weight ranges at COMPILE time so NPUW's bank stores zero-copy views of
    // raw-resident closures during evaluation instead of materializing a full host copy of the
    // weights (the closures already live in the shared buffer). Also set post-compile for the
    // consolidation path. Only meaningful when weights were relocated into shared buffers.
    if (!weight_ranges.empty()) {
        npu_props["XPU_RAW_WEIGHT_RANGES"] = weight_ranges;
    }

    // Auto-detect KV cache axes from ReadValue node shapes (like GenAI does).
    // For each 4D ReadValue input: dim with value 0 → seq_len, dynamic dim → batch.
    // Only set if not already provided by the application.
    if (npu_props.find("NPUW_LLM_BATCH_DIM") == npu_props.end() ||
        npu_props.find("NPUW_LLM_SEQ_LEN_DIM") == npu_props.end()) {
        uint64_t batch_dim = 0, seq_len_dim = 2;  // defaults
        for (const auto& op : model->get_ops()) {
            auto read_value = std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(op);
            if (!read_value || read_value->get_input_size() < 1)
                continue;
            auto shape = read_value->get_input_partial_shape(0);
            if (shape.rank().is_dynamic() || shape.rank().get_length() != 4)
                continue;
            for (int64_t i = 0; i < 4; i++) {
                if (shape[i] == ov::Dimension(0)) {
                    seq_len_dim = static_cast<uint64_t>(i);
                } else if (shape[i].is_dynamic()) {
                    batch_dim = static_cast<uint64_t>(i);
                }
            }
            break;  // use first 4D ReadValue
        }
        if (npu_props.find("NPUW_LLM_BATCH_DIM") == npu_props.end()) {
            npu_props["NPUW_LLM_BATCH_DIM"] = batch_dim;
        }
        if (npu_props.find("NPUW_LLM_SEQ_LEN_DIM") == npu_props.end()) {
            npu_props["NPUW_LLM_SEQ_LEN_DIM"] = seq_len_dim;
        }
        xpu_dbg() << "[XPU] step 6: auto-detected KV axes: batch_dim=" << batch_dim
                  << " seq_len_dim=" << seq_len_dim << std::endl;
    }

    // Step 6: Compile NPU WITHOUT shared buffer (deferred allocation).
    // NPUW loads full weights during compilation. Consolidation into shared buffer
    // happens via set_property() after compilation completes.
    auto npu_compiled = core->compile_model(model, "NPU", npu_props);
    xpu_dbg() << "[XPU] step 6: NPU compile OK" << std::endl;
    XPU_MEM("after NPU compile");

    // Step 6a': Register the GPU-shared raw weight buffer (the .bin) with NPUW. Plain
    //   Const closures that are byte-identical to it (the INT4 transformer weights, which
    //   NPUW keeps un-transformed) are then pointed AT the raw buffer (zero-copy) rather
    //   than copied into the persistent buffer — so GPU and NPU hold ONE physical copy.
    //   Must run before the size query below so the persistent buffer excludes them.
    npu_compiled->set_property({
        {"XPU_RAW_WEIGHT_RANGES", weight_ranges},
    });

    // Step 6b: Query exact evaluated weight size (guaranteed complete — eval waited in ctor).
    //   NPUW_WEIGHTS_TOTAL_BYTES   — Bank-managed closures NOT raw-resident (need copying)
    //   NPUW_HOST_CLOSURE_TOTAL_BYTES — host-side closures (scales/zero-points/biases/
    //     norms/embeddings) that bypass the Bank. Both must live in the shared buffer
    //     so every weight is cross-device accessible.
    auto bank_bytes = npu_compiled->get_property("NPUW_WEIGHTS_TOTAL_BYTES").as<uint64_t>();
    uint64_t host_bytes = 0;
    try {
        host_bytes = npu_compiled->get_property("NPUW_HOST_CLOSURE_TOTAL_BYTES").as<uint64_t>();
    } catch (const std::exception& e) {
        xpu_dbg() << "[XPU] step 6b: NPUW_HOST_CLOSURE_TOTAL_BYTES unavailable (" << e.what()
                  << "), host closures will NOT be in shared buffer" << std::endl;
    }
    auto total_bytes = bank_bytes + host_bytes;
    xpu_dbg() << "[XPU] step 6b: weight size = " << total_bytes << " bytes ("
              << (total_bytes / 1048576.0) << " MB) = bank " << (bank_bytes / 1048576.0)
              << " MB + host closures " << (host_bytes / 1048576.0) << " MB" << std::endl;

    // Step 6c: Allocate persistent buffer with exact size (bank + host closures)
    auto persistent_buf = SharedWeightBuffer::create_empty(gpu_ctx, static_cast<size_t>(total_bytes));
    xpu_dbg() << "[XPU] step 6c: persistent buf created, host_ptr=" << persistent_buf->host_ptr
              << " alloc_size=" << persistent_buf->alloc_size << std::endl;

    // Step 6d: Consolidate — set_property triggers bank->set_xpu_shared_buffer + consolidate
    npu_compiled->set_property({
        {"XPU_SHARED_WEIGHT_PTR", reinterpret_cast<uint64_t>(persistent_buf->host_ptr)},
        {"XPU_SHARED_WEIGHT_SIZE", static_cast<uint64_t>(persistent_buf->alloc_size)},
    });
    xpu_dbg() << "[XPU] step 6d: weights consolidated into exact-size buffer" << std::endl;

    // Step 6e: Allocate shared KV cache buffer (GPU USM) for GPU/NPU zero-copy KV sharing
    std::shared_ptr<SharedWeightBuffer> kvcache_buf;
    auto kv_bytes = npu_compiled->get_property("NPUW_KVCACHE_TOTAL_BYTES").as<uint64_t>();
    if (kv_bytes > 0) {
        kvcache_buf = SharedWeightBuffer::create_empty(gpu_ctx, static_cast<size_t>(kv_bytes));
        npu_compiled->set_property({
            {"XPU_SHARED_KVCACHE_PTR", reinterpret_cast<uint64_t>(kvcache_buf->host_ptr)},
            {"XPU_SHARED_KVCACHE_SIZE", static_cast<uint64_t>(kvcache_buf->alloc_size)},
        });
        xpu_dbg() << "[XPU] step 6e: KV cache buf created, host_ptr=" << kvcache_buf->host_ptr
                  << " size=" << kv_bytes << " alloc_size=" << kvcache_buf->alloc_size << std::endl;
    } else {
        xpu_dbg() << "[XPU] step 6e: no KV cache tensors found, skipping" << std::endl;
    }

    // Step 6f: Create stateless prefill model for GPU
    //   Clone the original (stateful) model, convert ReadValue/Assign to explicit
    //   past_key_values.*/present.* Parameters/Results via StatefulToStateless.
    //   This lets GPU produce explicit KV outputs that can be bridged to NPU's
    //   shared buffer.
    ov::SoPtr<ov::ICompiledModel> gpu_prefill_compiled;
    if (kvcache_buf) {
        xpu_dbg() << "[XPU] step 6f: creating stateless prefill model for GPU..." << std::endl;
        auto prefill_model = model->clone();
        ov::pass::StatefulToStateless().run_on_model(prefill_model);

        // GPU KV-write: rewrite present.* to the NPU's static KV slot layout so the GPU produces (and
        // writes zero-copy) the f16/transposed KV cache itself -- no host round-trip (§3.3).
        try {
            auto layout = npu_compiled->get_property("NPUW_KVCACHE_LAYOUT").as<std::string>();
            bool v_transposed = false;
            try {
                v_transposed = npu_compiled->get_property("NPUW_KVCACHE_V_TRANSPOSED_GEN").as<bool>();
            } catch (...) {
            }
            xpu_rewrite_present_to_static_slots(prefill_model, xpu_parse_kv_layout(layout), v_transposed);
        } catch (const std::exception& e) {
            // The hybrid REQUIRES the rewrite (there is no CPU-bridge fallback): a failed/partial rewrite
            // would leave the prefill model inconsistent with the KV-output bindings -> silently-wrong KV.
            // Fail the compile loudly instead.
            OPENVINO_THROW("[XPU] step 6f: GPU KV-write rewrite failed (required for the hybrid): ", e.what());
        }

        // Slice the prefill logits to the LAST token (port of GenAI apply_slice_before_matmul): the
        // prefill only needs logits[-1] for the first generated token, so shrink the LM-head matmul +
        // logits output from [1,seq,vocab] to [1,1,vocab] (drops the wasted full-seq LM head + the
        // ~622 MB logits copy at 1K).
        try {
            if (xpu_slice_prefill_logits_to_last(prefill_model))
                xpu_dbg() << "[XPU] step 6f: sliced prefill logits to last token (LM head -> 1 position)"
                          << std::endl;
            else
                xpu_dbg() << "[XPU] step 6f: prefill logits slice not applied (LM-head pattern unmatched)"
                          << std::endl;
        } catch (const std::exception& e) {
            xpu_dbg() << "[XPU] step 6f: prefill logits slice skipped (" << e.what() << ")" << std::endl;
        }
        // The clone shares the relocated Constants (which view the per-weight USM buffers),
        // so the same ranges apply — set them so GPU uses share_usm for those weights.
        // Share the buffer with GPU prefill only when it can actually consume it: USM weights, or
        // malloc weights that were imported into the GPU-L0 context (gpu_shares). Otherwise the GPU
        // copies its prefill weights (the NPU still zero-copy-imports the shared malloc buffer).
        if (gpu_shares) {
            prefill_model->set_rt_info(weight_ranges, "xpu_shared_weight_ranges");
        } else {
            xpu_dbg() << "[XPU] step 6f: GPU prefill copies weights (buffer not GPU-shareable); "
                         "NPU native-INT4 zero-copy-imports the shared buffer" << std::endl;
        }

        ov::AnyMap gpu_prefill_props;
        for (const auto& [key, val] : properties) {
            if (key.find("GPU") != std::string::npos) {
                gpu_prefill_props[key] = val;
            }
        }
        gpu_prefill_compiled = core->compile_model(prefill_model, "GPU", gpu_prefill_props);
        XPU_MEM("after GPU prefill compile");
        xpu_dbg() << "[XPU] step 6f: GPU prefill compile OK, inputs=" << gpu_prefill_compiled->inputs().size()
                  << " outputs=" << gpu_prefill_compiled->outputs().size() << std::endl;

        // Log present.* output names for debugging
        for (const auto& out : gpu_prefill_compiled->outputs()) {
            auto name = out.get_any_name();
            if (name.find("present") != std::string::npos) {
                xpu_dbg() << "[XPU] step 6f: GPU prefill output: " << name
                          << " " << out.get_partial_shape() << " " << out.get_element_type() << std::endl;
            }
        }
    } else {
        xpu_dbg() << "[XPU] step 6f: no KV cache, skipping GPU prefill model" << std::endl;
    }

    // 7. Compile the FULL stateful GPU model. This graph is ONLY used for GPU-only passthrough
    //    (XPU_ACTIVE_DEVICE=GPU). In hybrid mode (a stateless GPU prefill model was built at 6f)
    //    it is never executed, so skip it by default to save its compile time + GPU activation
    //    memory. Set XPU_COMPILE_GPU_FULL=1 to force-build it for passthrough use.
    ov::SoPtr<ov::ICompiledModel> gpu_compiled;
    const bool want_gpu_full = (std::getenv("XPU_COMPILE_GPU_FULL") != nullptr) || !gpu_prefill_compiled;
    if (want_gpu_full) {
        xpu_dbg() << "[XPU] step 7: compiling full stateful GPU model..." << std::endl;
        ov::AnyMap gpu_props;
        for (const auto& [key, val] : properties) {
            if (key.find("GPU") != std::string::npos) {
                gpu_props[key] = val;
            }
        }
        gpu_compiled = core->compile_model(model, "GPU", gpu_props);
        xpu_dbg() << "[XPU] step 7: GPU compile OK" << std::endl;
    } else {
        xpu_dbg() << "[XPU] step 7: skipped full stateful GPU compile (hybrid mode; "
                     "set XPU_COMPILE_GPU_FULL=1 for GPU-only passthrough)" << std::endl;
    }

    // 8. Return XPU compiled model wrapping both
    auto compiled = std::make_shared<XpuCompiledModel>(model,
                                                        shared_from_this(),
                                                        std::move(gpu_compiled),
                                                        std::move(npu_compiled),
                                                        std::move(gpu_prefill_compiled),
                                                        std::move(weight_segments),
                                                        std::move(persistent_buf),
                                                        std::move(kvcache_buf));

    // Apply XPU_ACTIVE_DEVICE if specified
    auto dev_it = properties.find("XPU_ACTIVE_DEVICE");
    if (dev_it != properties.end()) {
        compiled->set_property({{"XPU_ACTIVE_DEVICE", dev_it->second}});
    }

    XPU_MEM("after all compiles (final)");
    return compiled;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties) const {
    // For model-based compile, we need the model path from properties or model's path
    // Try to extract path from properties
    auto it = properties.find("model_path");
    if (it != properties.end()) {
        return compile_model(std::filesystem::path(it->second.as<std::string>()), properties);
    }

    // Fallback: compile on GPU and NPU directly with the given model (no shared buffer)
    auto core = get_core();
    OPENVINO_ASSERT(core, "[XPU] ICore is not set");

    auto npu_compiled = core->compile_model(model, "NPU");
    auto gpu_compiled = core->compile_model(model, "GPU");

    return std::make_shared<XpuCompiledModel>(model,
                                               shared_from_this(),
                                               std::move(gpu_compiled),
                                               std::move(npu_compiled),
                                               ov::SoPtr<ov::ICompiledModel>{},
                                               std::vector<std::shared_ptr<SharedWeightBuffer>>{},
                                               nullptr,
                                               nullptr);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties,
                                                           const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    // No XPU-specific properties for now
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
        };
    } else if (name == ov::device::full_name.name()) {
        return std::string("Intel XPU (GPU + NPU)");
    } else if (name == ov::available_devices.name()) {
        // Return a single device ID so CoreImpl::get_available_devices() lists "XPU"
        return std::vector<std::string>{""};
    } else if (name == ov::device::capabilities.name()) {
        return std::vector<std::string>{ov::device::capability::FP32, ov::device::capability::FP16,
                                         ov::device::capability::INT8};
    } else if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{};
    }
    OPENVINO_THROW("[XPU] Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                          const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                          const ov::SoPtr<ov::IRemoteContext>& context,
                                                          const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                          const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                          const ov::SoPtr<ov::IRemoteContext>& context,
                                                          const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                         const ov::AnyMap& properties) const {
    auto core = get_core();
    OPENVINO_ASSERT(core, "[XPU] ICore is not set");
    // Delegate to GPU for supported operations
    return core->query_model(model, "GPU", properties);
}

}  // namespace ov::intel_xpu

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_intel_xpu_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::intel_xpu::Plugin, version)
