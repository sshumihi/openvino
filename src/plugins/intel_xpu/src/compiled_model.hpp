// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node_output.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "shared_weight_buffer.hpp"

namespace ov::intel_xpu {

class XpuCompiledModel : public ov::ICompiledModel {
public:
    XpuCompiledModel(const std::shared_ptr<const ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     ov::SoPtr<ov::ICompiledModel> gpu_compiled,
                     ov::SoPtr<ov::ICompiledModel> npu_compiled,
                     ov::SoPtr<ov::ICompiledModel> gpu_prefill_compiled,
                     std::vector<std::shared_ptr<SharedWeightBuffer>> weight_segments,
                     std::shared_ptr<SharedWeightBuffer> persistent_weights,
                     std::shared_ptr<SharedWeightBuffer> kvcache_buffer);

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    const ov::SoPtr<ov::ICompiledModel>& get_gpu_compiled() const { return m_gpu_compiled; }
    const ov::SoPtr<ov::ICompiledModel>& get_npu_compiled() const { return m_npu_compiled; }
    const ov::SoPtr<ov::ICompiledModel>& get_gpu_prefill_compiled() const { return m_gpu_prefill_compiled; }
    const std::string& get_default_device() const { return m_default_device; }

    // GPU prefill present.* output -> shared-KV-slot remote tensor, created ONCE at compile (the costly
    // create_tensor) so the infer request only set_tensor's them once instead of rebuilding all 72 every
    // prefill. Valid because the present.* outputs are static (full max_kv slot, §3.3) -> the binding is
    // conversation- and prompt-length-invariant.
    struct KvOutputBinding {
        ov::Output<const ov::Node> port;
        ov::SoPtr<ov::ITensor> tensor;
    };
    const std::vector<KvOutputBinding>& get_kv_output_bindings() const { return m_kv_output_bindings; }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    // A representative sub-model for metadata/export/runtime-model/property fallback. In hybrid
    // mode the full GPU model (m_gpu_compiled) may be skipped, so fall back to the GPU prefill
    // model, then the NPU model.
    const ov::SoPtr<ov::ICompiledModel>& meta_compiled() const {
        if (m_gpu_compiled)
            return m_gpu_compiled;
        if (m_gpu_prefill_compiled)
            return m_gpu_prefill_compiled;
        return m_npu_compiled;
    }

private:
    // Destroyed last (declared first). Persistent buffer holds NPUW transformed
    // weights referenced by NPU Bank at runtime.
    std::shared_ptr<SharedWeightBuffer> m_persistent_weights;
    // KV cache GPU USM buffer shared between GPU and NPU.
    std::shared_ptr<SharedWeightBuffer> m_kvcache_buffer;
    // Per-weight GPU-USM-host buffers the model's Constants view (each <= 2GB so the GPU
    // can address them). Held for the compiled-model lifetime; both GPU (share_usm) and
    // NPU (host-unpack/raw-share) read from these.
    std::vector<std::shared_ptr<SharedWeightBuffer>> m_weight_segments;
    ov::SoPtr<ov::ICompiledModel> m_gpu_compiled;
    ov::SoPtr<ov::ICompiledModel> m_npu_compiled;
    ov::SoPtr<ov::ICompiledModel> m_gpu_prefill_compiled;  // Stateless GPU prefill model
    // Declared AFTER m_gpu_prefill_compiled / m_kvcache_buffer so it is destroyed FIRST: the remote
    // tensors reference the GPU context + the shared KV buffer, which must outlive them.
    std::vector<KvOutputBinding> m_kv_output_bindings;
    std::string m_default_device = "GPU";

    // Create the per-slot GPU remote tensors (compile-stage); called from the ctor.
    void build_kv_output_bindings();
};

}  // namespace ov::intel_xpu
