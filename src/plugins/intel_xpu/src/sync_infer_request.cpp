// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xpu_debug.hpp"
#include "sync_infer_request.hpp"

#include <algorithm>

#include "openvino/runtime/make_tensor.hpp"

namespace ov::intel_xpu {

XpuSyncInferRequest::XpuSyncInferRequest(const std::shared_ptr<const XpuCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(compiled_model),
      m_npu_request(compiled_model->get_npu_compiled()->create_infer_request()) {
    // The full stateful GPU model is only built for GPU-only passthrough; in hybrid mode it is
    // skipped, so create its request only when present.
    if (compiled_model->get_gpu_compiled()) {
        m_gpu_request = compiled_model->get_gpu_compiled()->create_infer_request();
    }
    // Create GPU prefill request if available
    if (compiled_model->get_gpu_prefill_compiled()) {
        m_gpu_prefill_request = compiled_model->get_gpu_prefill_compiled()->create_infer_request();

        // Bind the present.* outputs to the compile-stage shared-KV-slot remote tensors ONCE here, so
        // infer_gpu_prefill doesn't rebuild + set_tensor 72 remote tensors on every prefill. The outputs
        // are static (full max_kv slot, §3.3), so the binding holds across conversations / prompt lengths.
        const auto& kv_bindings = compiled_model->get_kv_output_bindings();
        for (const auto& b : kv_bindings) {
            m_gpu_prefill_request->set_tensor(b.port, b.tensor);
        }
        m_kv_outputs_bound = !kv_bindings.empty();
        if (m_kv_outputs_bound)
            xpu_dbg() << "[XPU] bound " << kv_bindings.size()
                      << " present.* outputs to shared KV slots at request init (compile-stage tensors)"
                      << std::endl;
    }

    // Allocate default tensors for all input/output ports so check_tensors() doesn't fail.
    auto alloc_placeholder = [](const ov::Output<const ov::Node>& port, ov::SoPtr<ov::ITensor>& tensor) {
        if (tensor)
            return;
        auto pshape = port.get_partial_shape();
        if (pshape.is_static()) {
            tensor = ov::make_tensor(port.get_element_type(), pshape.get_shape());
        } else if (pshape.rank().is_static()) {
            ov::Shape min_shape;
            for (const auto& dim : pshape) {
                min_shape.push_back(dim.is_static() ? dim.get_length() : 1);
            }
            tensor = ov::make_tensor(port.get_element_type(), min_shape);
        }
    };
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [&](ov::SoPtr<ov::ITensor>& tensor) { alloc_placeholder(input, tensor); });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [&](ov::SoPtr<ov::ITensor>& tensor) { alloc_placeholder(output, tensor); });
    }
}

void XpuSyncInferRequest::infer() {
    auto input_ids = get_tensor_ptr(get_inputs()[0]);
    auto seq_len = input_ids->get_shape()[1];

    // When the active device is forced to NPU, run the FULL NPUW pipeline
    // (prefill + decode) directly through the NPUW LLM request, bypassing GPU
    // prefill entirely. The NPUW LLM request internally routes seq_len>1 to its
    // prefill model and seq_len==1 to its generate model. This is the "pure NPUW
    // via XPU shared buffer" path: it validates weight-consolidation accuracy
    // independent of the GPU-prefill KV bridge.
    if (m_active_device == "NPU") {
        infer_npu_decode();
        return;
    }

    // A seq_len>1 infer is ALWAYS a (re)prefill -- a new conversation or a chat turn -- so route it to
    // the GPU prefill even after a prior prefill (do NOT latch on m_prefill_done). The previous
    // `!m_prefill_done` guard misrouted a 2nd conversation's prefill into infer_npu_decode(), feeding a
    // multi-token prompt to the NPU's single-token generate graph (crash). infer_gpu_prefill re-signals
    // the NPU's external-prefill length, so the reused NPU request re-initializes (num_stored_tokens,
    // generate variant) for the new conversation on its next decode.
    // Route to GPU prefill when (a) this is a multi-token prefill (a new conversation / chat turn -- always
    // (re)prefills, even after a prior prefill), or (b) it's the very first infer of a conversation (covers a
    // degenerate 1-token prompt, which still prefills rather than hitting the GPU-only passthrough that asserts
    // on the null m_gpu_request in hybrid mode). After a prefill, seq_len==1 infers are decode steps.
    if (m_gpu_prefill_request && (seq_len > 1 || !m_prefill_done)) {
        infer_gpu_prefill();
    } else if (m_prefill_done) {
        infer_npu_decode();
    } else {
        infer_passthrough();
    }
}

void XpuSyncInferRequest::infer_passthrough() {
    OPENVINO_ASSERT(m_active_device == "NPU" || m_gpu_request,
                    "[XPU] GPU-only passthrough requested but the full stateful GPU model was not "
                    "compiled (hybrid mode). Re-compile with XPU_COMPILE_GPU_FULL=1, or set "
                    "XPU_ACTIVE_DEVICE=NPU.");
    auto& active_request = (m_active_device == "NPU") ? m_npu_request : m_gpu_request;
    auto& active_compiled = (m_active_device == "NPU")
                                ? m_compiled_model->get_npu_compiled()
                                : m_compiled_model->get_gpu_compiled();

    auto active_inputs = active_compiled->inputs();
    size_t n_inputs = std::min(get_inputs().size(), active_inputs.size());
    for (size_t i = 0; i < n_inputs; ++i) {
        try {
            auto tensor = get_tensor_ptr(get_inputs()[i]);
            if (tensor) {
                active_request->set_tensor(active_inputs[i], tensor);
            }
        } catch (...) {
        }
    }

    active_request->infer();

    auto active_outputs = active_compiled->outputs();
    size_t n_outputs = std::min(get_outputs().size(), active_outputs.size());
    for (size_t i = 0; i < n_outputs; ++i) {
        try {
            auto tensor = active_request->get_tensor(active_outputs[i]);
            if (tensor) {
                ISyncInferRequest::set_tensor(get_outputs()[i], tensor);
            }
        } catch (...) {
        }
    }
}

void XpuSyncInferRequest::infer_gpu_prefill() {
    xpu_dbg() << "[XPU] infer_gpu_prefill: starting GPU prefill..." << std::endl;

    auto& prefill_compiled = m_compiled_model->get_gpu_prefill_compiled();
    auto prefill_inputs = prefill_compiled->inputs();
    auto prefill_outputs = prefill_compiled->outputs();

    auto input_ids = get_tensor_ptr(get_inputs()[0]);
    auto prompt_len = input_ids->get_shape()[1];

    // 1. Forward user inputs (input_ids, attention_mask, position_ids) by name
    for (const auto& xpu_input : get_inputs()) {
        auto xpu_names = xpu_input.get_names();
        for (const auto& prefill_input : prefill_inputs) {
            auto prefill_names = prefill_input.get_names();
            for (const auto& xn : xpu_names) {
                if (prefill_names.count(xn) > 0) {
                    auto tensor = get_tensor_ptr(xpu_input);
                    if (tensor) {
                        m_gpu_prefill_request->set_tensor(prefill_input, tensor);
                    }
                    goto next_xpu_input;
                }
            }
        }
        next_xpu_input:;
    }

    // 2. Set GPU's past_key_values.* inputs to empty (zero-length) tensors.
    //    The stateless model expects past_key_values with shape [batch, heads, 0, head_dim]
    //    for first prefill (no prior KV cache). Using 0 for the sequence dimension means
    //    the model's total sequence length equals prompt_len, matching the attention_mask.
    for (const auto& prefill_input : prefill_inputs) {
        auto name = prefill_input.get_any_name();
        if (name.find("past_key_values") == std::string::npos)
            continue;

        auto pshape = prefill_input.get_partial_shape();
        if (pshape.rank().is_static()) {
            ov::Shape empty_shape;
            for (size_t d = 0; d < pshape.rank().get_length(); d++) {
                if (pshape[d].is_static()) {
                    empty_shape.push_back(pshape[d].get_length());
                } else {
                    // Dynamic dimension: use 0 for the KV sequence length,
                    // 1 for batch (batch is always 1).
                    // For 4D KV [batch, heads, seq, head_dim], dim 2 is seq.
                    // For models with kv_dim=2, use 0 for dim 2, 1 for dim 0.
                    empty_shape.push_back(d == 2 ? 0 : 1);
                }
            }
            auto empty_tensor = ov::make_tensor(prefill_input.get_element_type(), empty_shape);
            m_gpu_prefill_request->set_tensor(prefill_input, empty_tensor);
        }
    }

    // The present.* outputs were rewritten to the NPU's static KV slot layout (plugin.cpp, §3.3) and bound
    // zero-copy to the shared KV buffer ONCE at request construction (compile-stage tensor cache), so the
    // GPU writes the NPU's KV cache directly during infer -- nothing to bind per prefill.
    OPENVINO_ASSERT(m_kv_outputs_bound,
                    "[XPU] GPU prefill present.* outputs were not bound to the shared KV buffer "
                    "(KV-write rewrite or compile-stage binding unavailable)");

    // Reuse the NPU generate request (built once at construction) and just SIGNAL this conversation's
    // prompt_len via the compiled-model property. The NPUW request consumes it at the top of its next
    // (decode) infer -- selecting the generate variant + setting num_stored_tokens. No pre-infer memset:
    // the rewritten graph's ScatterUpdate(Broadcast(0)) re-zeros the full KV slot during infer, overwriting
    // any stale KV from a prior conversation before NPU decode reads it.
    m_compiled_model->get_npu_compiled()._ptr->set_property(
        {{"XPU_EXTERNAL_PREFILL_LEN", static_cast<uint64_t>(prompt_len)}});
    xpu_dbg() << "[XPU] signaled NPU external prefill_len=" << prompt_len << " (consumed at next decode)"
              << std::endl;

    xpu_dbg() << "[XPU] running GPU prefill (KV-write directly to shared buffer)..." << std::endl;
    m_gpu_prefill_request->infer();

    // 6. Extract logits from GPU output
    for (const auto& prefill_output : prefill_outputs) {
        if (prefill_output.get_names().count("logits") > 0) {
            auto logits_tensor = m_gpu_prefill_request->get_tensor(prefill_output);
            for (const auto& xpu_output : get_outputs()) {
                if (xpu_output.get_names().count("logits") > 0) {
                    ISyncInferRequest::set_tensor(xpu_output, logits_tensor);
                    break;
                }
            }
            break;
        }
    }

    m_prefill_done = true;
}

void XpuSyncInferRequest::infer_npu_decode() {
    auto& npu_compiled = m_compiled_model->get_npu_compiled();

    // Forward input tensors by name to NPU
    auto npu_inputs = npu_compiled->inputs();
    for (const auto& xpu_input : get_inputs()) {
        auto xpu_names = xpu_input.get_names();
        for (const auto& npu_input : npu_inputs) {
            auto npu_names = npu_input.get_names();
            for (const auto& xn : xpu_names) {
                if (npu_names.count(xn) > 0) {
                    auto tensor = get_tensor_ptr(xpu_input);
                    if (tensor) {
                        m_npu_request->set_tensor(npu_input, tensor);
                    }
                    goto next_npu_input;
                }
            }
        }
        next_npu_input:;
    }

    m_npu_request->infer();

    // Extract logits from NPU output
    auto npu_outputs = npu_compiled->outputs();
    for (const auto& npu_output : npu_outputs) {
        if (npu_output.get_names().count("logits") > 0) {
            auto logits = m_npu_request->get_tensor(npu_output);
            for (const auto& xpu_output : get_outputs()) {
                if (xpu_output.get_names().count("logits") > 0) {
                    ISyncInferRequest::set_tensor(xpu_output, logits);
                    break;
                }
            }
            break;
        }
    }
}

std::vector<ov::SoPtr<ov::IVariableState>> XpuSyncInferRequest::query_state() const {
    // In hybrid mode the full stateful GPU model is skipped, so m_gpu_request is null -- fall back to the
    // NPU request (which exists) rather than null-deref before the first prefill.
    if (m_active_device == "NPU" || m_prefill_done || !m_gpu_request) {
        return m_npu_request->query_state();
    }
    return m_gpu_request->query_state();
}

std::vector<ov::ProfilingInfo> XpuSyncInferRequest::get_profiling_info() const {
    if (m_active_device == "NPU" || m_prefill_done || !m_gpu_request) {
        return m_npu_request->get_profiling_info();
    }
    return m_gpu_request->get_profiling_info();
}

void XpuSyncInferRequest::set_active_device(const std::string& device) {
    OPENVINO_ASSERT(device == "GPU" || device == "NPU",
                    "[XPU] Active device must be 'GPU' or 'NPU', got: ", device);
    m_active_device = device;
}

}  // namespace ov::intel_xpu
