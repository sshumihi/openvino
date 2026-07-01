// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "xpu_debug.hpp"

namespace ov {
namespace npuw {

/// RAII wrapper that owns ONE shared INT4 weight segment as a page-aligned host malloc that is
/// imported into BOTH the GPU-L0 and NPU-L0 contexts, so the GPU (share_usm) and the NPU
/// (native-INT4 import) read the SAME physical bytes — no per-device duplication.
///
/// Ported verbatim from the intel_xpu plugin (shared_weight_buffer.hpp) so the same
/// "one physical copy across GPU+NPU" mechanism lives INSIDE NPUW, making the separate
/// intel_xpu meta-plugin unnecessary for the hybrid GPU-prefill / NPU-decode case.
struct SharedWeightBuffer {
    void* host_ptr = nullptr;
    size_t alloc_size = 0;  // total allocated size

    SharedWeightBuffer() = default;
    SharedWeightBuffer(const SharedWeightBuffer&) = delete;
    SharedWeightBuffer& operator=(const SharedWeightBuffer&) = delete;

    // True if host_ptr was imported into the GPU's Level-Zero context: the page-aligned malloc is
    // registered in BOTH the GPU-L0 and NPU-L0 contexts as the SAME pointer, so GPU share_usm AND
    // NPU native-INT4 read one physical buffer (no copy).
    bool gpu_l0_imported = false;

    // True if host_ptr was imported into the NPU's Level-Zero context ONCE and is held resident for
    // this buffer's lifetime (see import_into_npu_l0). This keeps the buffer's ZeroMem alive in the
    // NPU mem pool, so the per-token decode weight import becomes a pool HIT (no zeMemAllocHost),
    // instead of being re-imported every token (the shared-weights decode regression).
    bool npu_l0_imported = false;

    ~SharedWeightBuffer() {
        // Release the NPU-L0 import (it references host_ptr) BEFORE freeing the malloc.
        m_npu_import = {};
        // Release the GPU-L0 import handle (same address as host_ptr) before freeing the malloc.
        if (m_ze_imported && m_ze_context) {
            zeMemFree(static_cast<ze_context_handle_t>(m_ze_context), m_ze_imported);
        }
        if (host_ptr) {
            _aligned_free(host_ptr);
        }
    }

    /// Import this buffer's malloc'd host_ptr into the GPU's Level-Zero context so the GPU can
    /// share_usm it (it must be a registered L0 allocation for zeMemGetAddressRange to succeed).
    /// Returns false (GPU then copies) on any failure. The NPU imports the same host_ptr
    /// independently via its own create_tensor at inference.
    bool import_into_gpu_l0(const ov::SoPtr<ov::IRemoteContext>& gpu_ctx) {
        if (!host_ptr || !gpu_ctx) {
            return false;
        }
        try {
            const auto& props = gpu_ctx->get_property();
            auto it = props.find("OCL_CONTEXT");  // == ze_context_handle_t on the L0 GPU backend
            if (it == props.end()) {
                return false;
            }
            auto ze_ctx = static_cast<ze_context_handle_t>(it->second.as<void*>());
            if (!ze_ctx) {
                return false;
            }

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
                if (r == ZE_RESULT_SUCCESS && imported) {
                    zeMemFree(ze_ctx, imported);
                }
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

    /// Import this buffer's malloc'd host_ptr into the NPU's Level-Zero context ONCE and HOLD the
    /// resulting remote tensor for this buffer's lifetime. The import registers the buffer's ZeroMem
    /// in the NPU mem pool; because we keep it alive, the per-token decode weight binding
    /// (set_tensor -> ZeroTensor -> import_standard_allocation_memory) resolves to a pool HIT instead
    /// of issuing a fresh zeMemAllocHost every token. This is what makes shared-weight NPU decode run
    /// at native speed (no per-token weight re-import). Requires the buffer size to be page-aligned
    /// (create_empty rounds it up). Returns false (decode then re-imports per token) on any failure.
    bool import_into_npu_l0(const ov::SoPtr<ov::IRemoteContext>& npu_ctx) {
        if (!host_ptr || !npu_ctx) {
            return false;
        }
        try {
            ov::AnyMap params = {
                {ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::CPU_VA},
                {ov::intel_npu::mem_handle.name(), host_ptr},
                {ov::intel_npu::tensor_type.name(), ov::intel_npu::TensorType::INPUT},
            };
            m_npu_import = npu_ctx._ptr->create_tensor(ov::element::u8, ov::Shape{alloc_size}, params);
            npu_l0_imported = (m_npu_import != nullptr);
            return npu_l0_imported;
        } catch (const std::exception& e) {
            ::ov::npuw::xpu_dbg() << "[NPUW] SharedWeightBuffer::import_into_npu_l0 failed (" << e.what()
                                  << "), NPU will re-import this weight per decode token" << std::endl;
            return false;
        }
    }

    /// Allocate a page-aligned host buffer and import it into the GPU-L0 context (Approach 3).
    /// `zero == false` leaves the buffer uninitialized (caller overwrites it fully via memcpy)
    /// to avoid a wasteful memset over multi-GB of weights.
    static std::shared_ptr<SharedWeightBuffer> create_empty(const ov::SoPtr<ov::IRemoteContext>& gpu_ctx,
                                                            size_t alloc_bytes,
                                                            bool zero = false) {
        OPENVINO_ASSERT(alloc_bytes > 0, "[NPUW] SharedWeightBuffer: alloc_bytes must be > 0");
        auto buf = std::make_shared<SharedWeightBuffer>();
        // Round the allocation up to a whole page so the buffer can also be imported into the NPU-L0
        // context (CPU_VA import requires BOTH the start AND the size to be page-aligned). The weight
        // occupies the first alloc_bytes; the padding is unused. Start is page-aligned via _aligned_malloc.
        const size_t alloc_rounded = (alloc_bytes + 4095u) & ~static_cast<size_t>(4095u);
        buf->alloc_size = alloc_rounded;
        buf->host_ptr = _aligned_malloc(alloc_rounded, 4096);
        OPENVINO_ASSERT(buf->host_ptr, "[NPUW] SharedWeightBuffer: failed to allocate ", alloc_rounded, " bytes");
        if (zero) {
            std::memset(buf->host_ptr, 0, alloc_rounded);
        }
        if (gpu_ctx) {
            buf->import_into_gpu_l0(gpu_ctx);
        }
        return buf;
    }

private:
    void* m_ze_context = nullptr;  // GPU L0 context the malloc was imported into
    void* m_ze_imported = nullptr;  // L0 import handle (== host_ptr); freed in dtor
    ov::SoPtr<ov::IRemoteTensor> m_npu_import;  // held NPU-L0 import; keeps the buffer pool-resident
};

}  // namespace npuw
}  // namespace ov
