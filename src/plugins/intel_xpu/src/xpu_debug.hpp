// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <streambuf>

namespace ov::intel_xpu {

/// True when OV_XPU_DEBUG is set in the environment (checked once).
inline bool xpu_debug_enabled() {
    static const bool on = (std::getenv("OV_XPU_DEBUG") != nullptr);
    return on;
}

/// Debug stream for the plugin's "[XPU] ..." progress prints. Returns std::cout when
/// OV_XPU_DEBUG is set, otherwise a sink that discards everything written to it. Use as a
/// drop-in for std::cout: `xpu_dbg() << "[XPU] ..." << std::endl;`.
inline std::ostream& xpu_dbg() {
    struct NullBuffer : std::streambuf {
        int overflow(int c) override { return c; }  // discard, report success
    };
    static NullBuffer null_buf;
    static std::ostream null_stream(&null_buf);
    return xpu_debug_enabled() ? std::cout : null_stream;
}

}  // namespace ov::intel_xpu
