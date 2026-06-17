// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <streambuf>

namespace ov {
namespace npuw {

/// True when OV_XPU_DEBUG is set in the environment (checked once). Shared gate for the
/// XPU-fork's "[NPUW ...]" / "[DCOFF]" / "[DIAG]" progress prints so they are silent by default.
inline bool xpu_debug_enabled() {
    static const bool on = (std::getenv("OV_XPU_DEBUG") != nullptr);
    return on;
}

/// Drop-in for std::cout/std::cerr: returns the real stream when OV_XPU_DEBUG is set, else a
/// sink that discards. Use as `xpu_dbg() << "[NPUW] ..." << std::endl;`.
inline std::ostream& xpu_dbg() {
    struct NullBuffer : std::streambuf {
        int overflow(int c) override { return c; }  // discard, report success
    };
    static NullBuffer null_buf;
    static std::ostream null_stream(&null_buf);
    return xpu_debug_enabled() ? std::cout : null_stream;
}

}  // namespace npuw
}  // namespace ov
