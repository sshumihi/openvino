// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <regex>
#include <sstream>
#include <unordered_map>

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/properties.hpp"
#include "sync_infer_request.hpp"
#include "xpu_debug.hpp"

namespace ov::intel_xpu {

XpuCompiledModel::XpuCompiledModel(const std::shared_ptr<const ov::Model>& model,
                                     const std::shared_ptr<const ov::IPlugin>& plugin,
                                     ov::SoPtr<ov::ICompiledModel> gpu_compiled,
                                     ov::SoPtr<ov::ICompiledModel> npu_compiled,
                                     ov::SoPtr<ov::ICompiledModel> gpu_prefill_compiled,
                                     std::vector<std::shared_ptr<SharedWeightBuffer>> weight_segments,
                                     std::shared_ptr<SharedWeightBuffer> persistent_weights,
                                     std::shared_ptr<SharedWeightBuffer> kvcache_buffer)
    : ov::ICompiledModel(model, plugin),
      m_persistent_weights(std::move(persistent_weights)),
      m_kvcache_buffer(std::move(kvcache_buffer)),
      m_weight_segments(std::move(weight_segments)),
      m_gpu_compiled(std::move(gpu_compiled)),
      m_npu_compiled(std::move(npu_compiled)),
      m_gpu_prefill_compiled(std::move(gpu_prefill_compiled)) {
    build_kv_output_bindings();
}

void XpuCompiledModel::build_kv_output_bindings() {
    // Pre-create (compile-stage) the GPU remote tensors that wrap each present.* output's slot in the
    // shared KV buffer, so the infer request binds them ONCE (set_tensor) at construction instead of
    // rebuilding all 72 every prefill. The present.* outputs are static (full max_kv slot, §3.3), so the
    // binding is conversation-invariant. The infer request asserts the bindings exist before prefill.
    if (!m_gpu_prefill_compiled || !m_npu_compiled || !m_kvcache_buffer || !m_kvcache_buffer->host_ptr)
        return;

    std::string layout;
    try {
        layout = m_npu_compiled->get_property("NPUW_KVCACHE_LAYOUT").as<std::string>();
    } catch (const std::exception&) {
        return;
    }
    struct Slot {
        uint64_t offset = 0;
        ov::Shape shape;
    };
    std::unordered_map<std::string, Slot> slots;
    std::istringstream ss(layout);
    std::string entry;
    while (std::getline(ss, entry, ';')) {
        if (entry.empty())
            continue;
        std::istringstream es(entry);
        std::string name, off_str, type_str, shape_str;
        std::getline(es, name, ':');
        std::getline(es, off_str, ':');
        std::getline(es, type_str, ':');
        std::getline(es, shape_str, ':');
        Slot slot;
        try {
            slot.offset = std::stoull(off_str);
        } catch (const std::exception&) {
            continue;
        }
        std::istringstream shape_ss(shape_str);
        std::string dim;
        while (std::getline(shape_ss, dim, ','))
            slot.shape.push_back(std::stoull(dim));
        slots.emplace(std::move(name), std::move(slot));
    }
    if (slots.empty())
        return;

    auto* kv_ptr = reinterpret_cast<uint8_t*>(m_kvcache_buffer->host_ptr);
    ov::SoPtr<ov::IRemoteContext> gpu_ctx;
    try {
        gpu_ctx = m_gpu_prefill_compiled->get_context();
    } catch (const std::exception&) {
        return;
    }

    for (const auto& out : m_gpu_prefill_compiled->outputs()) {
        const auto out_name = out.get_any_name();
        if (out_name.find("present") == std::string::npos)
            continue;
        const std::string buf_name = std::regex_replace(out_name, std::regex("present"), "past_key_values");
        auto it = slots.find(buf_name);
        if (it == slots.end())
            continue;
        // The KV-write rewrite (plugin.cpp step 6f) must have made this present.* output STATIC (the NPUW
        // slot shape). If it's still dynamic, the rewrite did not apply to it -> do NOT bind a static slot
        // tensor onto a dynamic port (that would silently mis-bind). Leave m_kv_output_bindings empty so the
        // request's OPENVINO_ASSERT(m_kv_outputs_bound) fails loudly instead.
        if (!out.get_partial_shape().is_static()) {
            xpu_dbg() << "[XPU] KV bind-cache: present output '" << out_name
                      << "' is not static (KV-write rewrite did not apply); aborting bind cache" << std::endl;
            m_kv_output_bindings.clear();
            return;
        }
        ov::AnyMap params = {
            {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
            {ov::intel_gpu::mem_handle.name(),
             static_cast<ov::intel_gpu::gpu_handle_param>(kv_ptr + it->second.offset)},
        };
        try {
            auto tensor = gpu_ctx->create_tensor(ov::element::f16, it->second.shape, params);
            m_kv_output_bindings.push_back({out, tensor});
        } catch (const std::exception& e) {
            // No per-prefill fallback exists: clearing the bindings disables GPU prefill (the request's
            // OPENVINO_ASSERT(m_kv_outputs_bound) will then fail loudly) rather than silently mis-binding.
            xpu_dbg() << "[XPU] KV bind-cache: create_tensor failed for " << out_name << ": " << e.what()
                      << " (GPU prefill will assert)" << std::endl;
            m_kv_output_bindings.clear();
            return;
        }
    }
    xpu_dbg() << "[XPU] KV bind-cache: pre-created " << m_kv_output_bindings.size()
              << " shared-KV-slot remote tensors at compile" << std::endl;
}

std::shared_ptr<ov::ISyncInferRequest> XpuCompiledModel::create_sync_infer_request() const {
    auto req = std::make_shared<XpuSyncInferRequest>(std::static_pointer_cast<const XpuCompiledModel>(shared_from_this()));
    req->set_active_device(m_default_device);
    return req;
}

void XpuCompiledModel::export_model(std::ostream& model) const {
    meta_compiled()->export_model(model);
}

std::shared_ptr<const ov::Model> XpuCompiledModel::get_runtime_model() const {
    return meta_compiled()->get_runtime_model();
}

void XpuCompiledModel::set_property(const ov::AnyMap& properties) {
    auto it = properties.find("XPU_ACTIVE_DEVICE");
    if (it != properties.end()) {
        auto device = it->second.as<std::string>();
        OPENVINO_ASSERT(device == "GPU" || device == "NPU",
                        "[XPU] XPU_ACTIVE_DEVICE must be 'GPU' or 'NPU', got: ", device);
        m_default_device = device;
    }
}

ov::Any XpuCompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},
        };
    } else if (name == ov::device::full_name.name()) {
        return std::string("Intel XPU (GPU + NPU)");
    } else if (name == ov::optimal_number_of_infer_requests.name()) {
        return static_cast<uint32_t>(1);
    }
    if (name == "XPU_ACTIVE_DEVICE") {
        return m_default_device;
    }
    if (name == "XPU_SHARED_WEIGHT_PTR") {
        return m_persistent_weights ? reinterpret_cast<uint64_t>(m_persistent_weights->host_ptr) : uint64_t(0);
    }
    if (name == "XPU_SHARED_WEIGHT_SIZE") {
        return m_persistent_weights ? static_cast<uint64_t>(m_persistent_weights->alloc_size) : uint64_t(0);
    }
    if (name == "XPU_WEIGHT_SEGMENT_COUNT") {
        return static_cast<uint64_t>(m_weight_segments.size());
    }
    if (name == "XPU_WEIGHT_SEGMENT_BYTES") {
        uint64_t total = 0;
        for (const auto& s : m_weight_segments)
            if (s) total += s->alloc_size;
        return total;
    }
    if (name == "XPU_WEIGHT_RANGES") {
        // "ptr:size;ptr:size;..." for each per-weight shared buffer. Lets a host-side test poke a
        // real transformer weight in the shared buffer and observe both engines read it (one copy).
        std::ostringstream ss;
        for (const auto& s : m_weight_segments)
            if (s)
                ss << reinterpret_cast<uint64_t>(s->host_ptr) << ":" << s->alloc_size << ";";
        return ss.str();
    }
    if (name == "XPU_VERIFY_GPU_L0") {
        // L0-level proof that the GPU does NOT hold a device copy of the weights: query the GPU's
        // Level-Zero context for each shared weight pointer and confirm it maps the SAME host malloc
        // (zeMemGetAddressRange base == host_ptr; zeMemGetAllocProperties type == HOST/HOST_IMPORTED).
        std::ostringstream ss;
        if (!m_gpu_prefill_compiled) {
            return std::string("no GPU prefill model (XPU_VERIFY_GPU_L0 needs the hybrid GPU context)");
        }
        try {
            auto ctx = m_gpu_prefill_compiled->get_context();
            auto props_map = ctx->get_property();
            auto it = props_map.find("OCL_CONTEXT");  // == ze_context_handle_t on the L0 GPU backend
            if (it == props_map.end())
                return std::string("GPU context has no OCL_CONTEXT (not an L0 build?)");
            auto ze_ctx = static_cast<ze_context_handle_t>(it->second.as<void*>());
            size_t total = 0, base_match = 0, host_type = 0, device_type = 0;
            for (const auto& s : m_weight_segments) {
                if (!s || !s->host_ptr)
                    continue;
                ++total;
                void* base = nullptr;
                size_t sz = 0;
                if (zeMemGetAddressRange(ze_ctx, s->host_ptr, &base, &sz) == ZE_RESULT_SUCCESS &&
                    base == s->host_ptr)
                    ++base_match;
                ze_memory_allocation_properties_t mp{};
                mp.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
                ze_device_handle_t dev = nullptr;
                if (zeMemGetAllocProperties(ze_ctx, s->host_ptr, &mp, &dev) == ZE_RESULT_SUCCESS) {
                    if (mp.type == ZE_MEMORY_TYPE_HOST || mp.type == ZE_MEMORY_TYPE_HOST_IMPORTED)
                        ++host_type;
                    else if (mp.type == ZE_MEMORY_TYPE_DEVICE)
                        ++device_type;
                }
            }
            ss << base_match << "/" << total << " segments: GPU-L0 base==host_ptr; " << host_type << "/" << total
               << " type=HOST(imported); " << device_type << " type=DEVICE";
            // Also audit the shared KV-cache buffer (same malloc + dual-L0-import path as the weights).
            if (m_kvcache_buffer && m_kvcache_buffer->host_ptr) {
                void* base = nullptr;
                size_t sz = 0;
                bool same = (zeMemGetAddressRange(ze_ctx, m_kvcache_buffer->host_ptr, &base, &sz) == ZE_RESULT_SUCCESS &&
                             base == m_kvcache_buffer->host_ptr);
                ze_memory_allocation_properties_t mp{};
                mp.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
                ze_device_handle_t dev = nullptr;
                zeMemGetAllocProperties(ze_ctx, m_kvcache_buffer->host_ptr, &mp, &dev);
                const char* t = (mp.type == ZE_MEMORY_TYPE_HOST || mp.type == ZE_MEMORY_TYPE_HOST_IMPORTED) ? "HOST"
                                : (mp.type == ZE_MEMORY_TYPE_DEVICE ? "DEVICE" : "OTHER");
                ss << " | KV buffer (" << (m_kvcache_buffer->alloc_size / 1048576) << " MB): base==ptr="
                   << same << " type=" << t;
            }
        } catch (const std::exception& e) {
            ss << "error: " << e.what();
        }
        return ss.str();
    }
    if (name == "XPU_SHARED_KVCACHE_PTR") {
        return m_kvcache_buffer ? reinterpret_cast<uint64_t>(m_kvcache_buffer->host_ptr) : uint64_t(0);
    }
    if (name == "XPU_SHARED_KVCACHE_SIZE") {
        return m_kvcache_buffer ? static_cast<uint64_t>(m_kvcache_buffer->alloc_size) : uint64_t(0);
    }
    if (name == "XPU_GPU_PREFILL_AVAILABLE") {
        return m_gpu_prefill_compiled != nullptr;
    }
    // Forward NPUW properties to the NPU compiled model
    if (name.find("NPUW") != std::string::npos && m_npu_compiled) {
        return m_npu_compiled->get_property(name);
    }
    // Delegate to a representative sub-model for other properties (full GPU model may be
    // skipped in hybrid mode).
    return meta_compiled()->get_property(name);
}

}  // namespace ov::intel_xpu
