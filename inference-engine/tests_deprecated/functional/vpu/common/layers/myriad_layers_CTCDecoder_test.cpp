// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_CTCDecoder_test.hpp"

INSTANTIATE_TEST_CASE_P(myriad, myriadCTCDecoderLayerTests_nightly,
        ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::ValuesIn(s_DimsConfig)));
