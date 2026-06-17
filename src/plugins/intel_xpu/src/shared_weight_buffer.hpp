// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "xpu_debug.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <ze_api.h>

#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov::intel_xpu {

/// RAII wrapper that owns the shared weight buffer.
/// Two allocation modes:
///   1) GPU-context USM: allocated via GPU's RemoteContext::create_host_tensor().
///      The GPU recognizes this as valid USM for zero-copy share_usm().
///   2) Fallback: plain _aligned_malloc host buffer (no GPU zero-copy).
struct SharedWeightBuffer {
    void* host_ptr = nullptr;
    size_t size = 0;        // .bin file size (raw weights), or 0 for empty buffers
    size_t alloc_size = 0;  // Total allocated size

    SharedWeightBuffer() = default;
    SharedWeightBuffer(const SharedWeightBuffer&) = delete;
    SharedWeightBuffer& operator=(const SharedWeightBuffer&) = delete;

    // True if host_ptr was imported into the GPU's Level-Zero context (POC "Approach 3"):
    // the page-aligned malloc is registered in BOTH the GPU-L0 and NPU-L0 contexts as the SAME
    // pointer, so GPU share_usm AND NPU native-INT4 read one physical buffer (no copy).
    bool gpu_l0_imported = false;

    ~SharedWeightBuffer() {
        // Release the GPU-L0 import handle (same address as host_ptr) before freeing the malloc.
        if (m_ze_imported && m_ze_context) {
            zeMemFree(static_cast<ze_context_handle_t>(m_ze_context), m_ze_imported);
        }
        // If using GPU host tensor, it owns the memory — don't free host_ptr
        if (!m_gpu_host_tensor && host_ptr) {
            _aligned_free(host_ptr);
        }
    }

    /// Import this buffer's malloc'd host_ptr into the GPU's Level-Zero context so the GPU can
    /// share_usm it (it must be a registered L0 allocation for zeMemGetAddressRange to succeed).
    /// Only valid for the malloc path (m_gpu_host_tensor == null) and a L0-backed GPU context on
    /// a platform that supports system-memory import (LNL/PTL). Returns false (GPU then copies)
    /// on any failure. The NPU imports the same host_ptr independently via its own create_tensor.
    bool import_into_gpu_l0(const ov::SoPtr<ov::IRemoteContext>& gpu_ctx) {
        if (!host_ptr || m_gpu_host_tensor || !gpu_ctx)
            return false;
        try {
            const auto& props = gpu_ctx->get_property();
            auto it = props.find("OCL_CONTEXT");  // == ze_context_handle_t on the L0 GPU backend
            if (it == props.end())
                return false;
            auto ze_ctx = static_cast<ze_context_handle_t>(it->second.as<void*>());
            if (!ze_ctx)
                return false;

            ze_external_memmap_sysmem_ext_desc_t sysmem{};
            sysmem.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC;
            sysmem.pSystemMemory = host_ptr;
            sysmem.size = alloc_size;

            ze_host_mem_alloc_desc_t hdesc{};
            hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
            hdesc.pNext = &sysmem;
            hdesc.flags = 0;

            void* imported = nullptr;
            ze_result_t r = zeMemAllocHost(ze_ctx, &hdesc, alloc_size, 4096, &imported);
            if (r != ZE_RESULT_SUCCESS || imported != host_ptr) {
                if (r == ZE_RESULT_SUCCESS && imported)
                    zeMemFree(ze_ctx, imported);
                return false;
            }
            m_ze_context = ze_ctx;
            m_ze_imported = imported;
            gpu_l0_imported = true;
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    /// Read the .bin file into a GPU-context USM host buffer for zero-copy weight sharing.
    /// If gpu_ctx is null, falls back to _aligned_malloc (no GPU zero-copy).
    /// Allocates exactly the .bin file size (size == alloc_size).
    static std::shared_ptr<SharedWeightBuffer> create(const std::string& bin_path,
                                                       const ov::SoPtr<ov::IRemoteContext>& gpu_ctx = {}) {
        auto buf = std::make_shared<SharedWeightBuffer>();

        // Read file size
        std::ifstream bin(bin_path, std::ios::binary | std::ios::ate);
        OPENVINO_ASSERT(bin.is_open(), "[XPU] Cannot open weights file: ", bin_path);
        buf->size = static_cast<size_t>(bin.tellg());
        OPENVINO_ASSERT(buf->size > 0, "[XPU] Weights file is empty: ", bin_path);
        bin.seekg(0, std::ios::beg);

        buf->alloc_size = buf->size;

        if (gpu_ctx) {
            // Allocate via GPU context — this creates USM host memory recognized by the GPU
            try {
                buf->m_gpu_host_tensor = gpu_ctx->create_host_tensor(ov::element::u8, {buf->alloc_size});
                buf->host_ptr = buf->m_gpu_host_tensor->data();
                OPENVINO_ASSERT(buf->host_ptr, "[XPU] GPU host tensor returned null data pointer");
                bin.read(static_cast<char*>(buf->host_ptr), buf->size);
                OPENVINO_ASSERT(bin.good(), "[XPU] Failed to read weights file: ", bin_path);
                bin.close();
                return buf;
            } catch (const std::exception& e) {
                // Fall through to aligned_malloc path
                xpu_dbg() << "[XPU] GPU host tensor alloc failed (" << e.what()
                          << "), falling back to _aligned_malloc" << std::endl;
                buf->m_gpu_host_tensor = {};
            }
        }

        // Fallback: allocate 4096-aligned host buffer
        buf->host_ptr = _aligned_malloc(buf->alloc_size, 4096);
        OPENVINO_ASSERT(buf->host_ptr, "[XPU] Failed to allocate ", buf->alloc_size, " bytes for weights");
        bin.read(static_cast<char*>(buf->host_ptr), buf->size);
        OPENVINO_ASSERT(bin.good(), "[XPU] Failed to read weights file: ", bin_path);
        bin.close();
        return buf;
    }

    /// Allocate an empty GPU USM host buffer (no file loading).
    /// Used for the persistent buffer and per-weight relocation segments.
    /// size = 0 (no bin data), alloc_size = alloc_bytes. When `zero` is false the
    /// buffer is left uninitialized (caller overwrites it fully, e.g. via memcpy) to
    /// avoid a wasteful memset over multi-GB of weights.
    static std::shared_ptr<SharedWeightBuffer> create_empty(
        const ov::SoPtr<ov::IRemoteContext>& gpu_ctx, size_t alloc_bytes, bool zero = true) {
        OPENVINO_ASSERT(alloc_bytes > 0, "[XPU] create_empty: alloc_bytes must be > 0");
        auto buf = std::make_shared<SharedWeightBuffer>();
        buf->size = 0;
        buf->alloc_size = alloc_bytes;

        // DEFAULT = page-aligned _aligned_malloc host buffer (imported into both L0 contexts). The NPU
        // Level-Zero backend can IMPORT a plain page-aligned host allocation (POC "Approach 3") — required
        // for native-INT4 execution to read weights from the shared buffer — but CANNOT import an OCL-USM
        // allocation ("part of an existing allocation"). Opt out to OCL USM-host with XPU_USM_WEIGHTS=1
        // (legacy DCOFF concept-proof path, where the NPU reads weights host-side and never imports them).
        const bool force_malloc = (std::getenv("XPU_USM_WEIGHTS") == nullptr);

        if (gpu_ctx && !force_malloc) {
            try {
                buf->m_gpu_host_tensor = gpu_ctx->create_host_tensor(ov::element::u8, {alloc_bytes});
                buf->host_ptr = buf->m_gpu_host_tensor->data();
                OPENVINO_ASSERT(buf->host_ptr, "[XPU] GPU host tensor returned null data pointer");
                if (zero)
                    std::memset(buf->host_ptr, 0, alloc_bytes);
                return buf;
            } catch (const std::exception& e) {
                xpu_dbg() << "[XPU] GPU host tensor alloc failed for empty buffer (" << e.what()
                          << "), falling back to _aligned_malloc" << std::endl;
                buf->m_gpu_host_tensor = {};
            }
        }

        // Fallback / forced: allocate 4096-aligned host buffer
        buf->host_ptr = _aligned_malloc(alloc_bytes, 4096);
        OPENVINO_ASSERT(buf->host_ptr, "[XPU] Failed to allocate ", alloc_bytes, " bytes for empty buffer");
        if (zero)
            std::memset(buf->host_ptr, 0, alloc_bytes);
        // POC "Approach 3": when forced to malloc for NPU L0 import, also import into the GPU's
        // L0 context so the GPU can share_usm the SAME physical buffer (one copy for both engines).
        if (force_malloc && gpu_ctx)
            buf->import_into_gpu_l0(gpu_ctx);
        return buf;
    }

    /// Returns true if the buffer was allocated via GPU context (USM host).
    bool is_gpu_usm() const { return m_gpu_host_tensor != nullptr; }

private:
    ov::SoPtr<ov::ITensor> m_gpu_host_tensor;  // Keeps GPU USM host memory alive
    void* m_ze_context = nullptr;              // GPU L0 context the malloc was imported into
    void* m_ze_imported = nullptr;             // L0 import handle (== host_ptr); freed in dtor
};

}  // namespace ov::intel_xpu
