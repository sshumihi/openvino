// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace ov::intel_xpu {

class XpuSyncInferRequest : public ov::ISyncInferRequest {
public:
    explicit XpuSyncInferRequest(const std::shared_ptr<const XpuCompiledModel>& compiled_model);

    void infer() override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    /// Switch the active device for inference ("GPU" or "NPU").
    void set_active_device(const std::string& device);
    const std::string& get_active_device() const { return m_active_device; }

private:
    /// Original passthrough logic: forward inputs/outputs to active device.
    void infer_passthrough();

    /// GPU prefill: run stateless GPU model, KV outputs → shared buffer, init NPU.
    void infer_gpu_prefill();

    /// NPU decode: forward to NPU LLMInferRequest for token-by-token generation.
    void infer_npu_decode();

    std::shared_ptr<const XpuCompiledModel> m_compiled_model;
    std::shared_ptr<ov::IAsyncInferRequest> m_gpu_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_npu_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_gpu_prefill_request;  // Stateless GPU prefill
    std::string m_active_device = "GPU";

    // GPU prefill → NPU decode orchestration state
    bool m_prefill_done = false;
    // True once the GPU prefill's present.* outputs are bound (in the ctor) to the shared-KV-slot remote
    // tensors created at compile (XpuCompiledModel::build_kv_output_bindings). infer_gpu_prefill asserts it.
    bool m_kv_outputs_bound = false;
};

}  // namespace ov::intel_xpu
