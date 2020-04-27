﻿// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "lrn_kernel_within_channel_ref.h"

namespace kernel_selector {
ParamsKey LRNKernelWithinChannel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLRNMode(LRNMode::WITHIN_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData LRNKernelWithinChannel::SetDefault(const lrn_params& params) const {
    CommonDispatchData runInfo = LRNKernelBase::SetDefault(params);

    runInfo.gws0 = 128 * 128;
    runInfo.gws1 = 1;
    runInfo.gws2 = 1;

    runInfo.lws0 = 128;
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants LRNKernelWithinChannel::GetJitConstants(const lrn_params& params,
                                                     const LRNKernelWithinChannel::Parent::DispatchData& kd) const {
    JitConstants jit = Parent::GetJitConstants(params, kd);
    const auto& input_dt = params.inputs[0].GetDType();

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"batch_id", "feature_id", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool LRNKernelWithinChannel::Validate(const Params& p, const optional_params& o) const {
    if (!LRNKernelBase::Validate(p, o)) {
        return false;
    }
    return true;
}

KernelsData LRNKernelWithinChannel::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_9);
}
}  // namespace kernel_selector
