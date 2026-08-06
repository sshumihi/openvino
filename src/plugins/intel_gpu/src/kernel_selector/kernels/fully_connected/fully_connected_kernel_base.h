// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_kernel_base.h"
#include "fully_connected_params.h"
#include <cstdlib>
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FullyConnectedKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class FullyConnectedKernelBase : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    using FusedOpDesc = fused_operation_desc;
    ~FullyConnectedKernelBase() override {}

    struct DispatchData : public CommonDispatchData {
        uint32_t unit_byte_size = 0;
        const char* chunk_type;
        uint32_t chunk_byte_size = 0;
        uint32_t units_per_chunk = 0;
        uint32_t bytes_per_sg_read = 0;
        uint32_t units_per_sg_read = 0;
        uint32_t responses_per_sg_exec = 0;
        uint32_t in_chunk_prefetch_size = 0;
        uint32_t filter_chunk_prefetch_size = 0;

        uint32_t last_rg_size = 0;
        uint32_t rg_count = 0;

        bool use_slm = false;
        uint32_t outer_n = 0;

        // Gemm style params
        uint32_t tile_m = 0;
        uint32_t tile_n = 0;
        uint32_t tile_mk = 0;
        uint32_t tile_nk = 0;
        uint32_t tile_ms = 0;
        uint32_t tile_ns = 0;
    };

    std::string GetAutoTuneOptions(int autoTuneIndex) const;
    std::vector<std::string> autoTuneOptions = {EXE_MODE_DEFAULT, EXE_MODE_NO_PRERA_SCH, EXE_MODE_AGE_BASED};
    using WeightBiasKernelBase::GetTunedKernelsDataByIndex;
    virtual KernelsData GetTunedKernelsDataByIndex(const Params &params,
                                                   DataLayout dl,
                                                   WeightsLayout wl,
                                                   const int autoTuneIndex = -1) const;

protected:
    using WeightBiasKernelBase::GetJitConstants;
    virtual JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const;
    virtual DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1, int kernel_number = 0) const;
    KernelsData GetCommonKernelsData(const Params &params,
                                     DataLayout dl,
                                     WeightsLayout wl,
                                     const std::string exeMode = EXE_MODE_DEFAULT,
                                     int autoTuneIndex = -1,
                                     int kernel_number = 0) const;

    // Fused ops
    virtual JitConstants GetFusedPrimitivesJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const;
    Datatype GetAccumulatorType(const fully_connected_params& params) const;
    Datatype GetActivationType(const fully_connected_params& params) const;
    // --Fused ops

    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    // Cross-plugin weight sharing: the tuned INT4/UINT4 FC kernels require a blocked weight layout,
    // so the plugin repacks the weight at compile time and a private device copy appears even when
    // the constant was imported zero-copy from a shared host bank. Declining those kernels makes the
    // selector fall back to FullyConnected_bfyx_Ref, which reads plain oiyx and needs no reorder.
    //
    // This switch is process-global and therefore a measurement aid, not a shippable option: it
    // declines the tuned kernel for *every* compressed i4 FC, including models with no shared
    // weights at all. The per-weight form is tractable — ProgramBuilder::remote_constant_ids
    // already records exactly which constants were imported — but needs that flag plumbed from the
    // plugin layer into fully_connected_params. Naming a real option is a review question.
    static bool decline_blocked_i4_for_shared_weights(const fully_connected_params& fc_params) {
        static const bool enabled = (std::getenv("OV_SHARED_WEIGHTS_NO_REPACK") != nullptr);
        return enabled && fc_params.compressed &&
               (fc_params.weights.GetDType() == WeightsType::INT4 ||
                fc_params.weights.GetDType() == WeightsType::UINT4);
    }
};
}  // namespace kernel_selector
