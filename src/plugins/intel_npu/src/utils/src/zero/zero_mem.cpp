// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mem.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

namespace {
// WI-2026-014 step 0 census. Separates level-zero memory that this process allocates fresh from
// memory that it imports from an existing host buffer, and names every refused import.
// A refused import falls back to a fresh allocation plus a copy, which is a second body.
// Enabled only by NPUW_MEM_CENSUS=1, so an unset run is byte-for-byte the old behaviour.
// Writes to stderr, because NPUW logging at DEBUG moves RSS itself (K-OPT-005).
bool census_on() {
    static const bool on = [] {
        const char* e = std::getenv("NPUW_MEM_CENSUS");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

struct CensusTotals {
    std::atomic<size_t> alloc_count{0}, alloc_bytes{0};
    std::atomic<size_t> import_count{0}, import_bytes{0};
    std::atomic<size_t> refused_count{0}, refused_bytes{0};
};

CensusTotals& totals() {
    static CensusTotals t;
    return t;
}

void census_alloc(size_t bytes) {
    if (!census_on()) {
        return;
    }
    const auto n = totals().alloc_count.fetch_add(1) + 1;
    const auto b = totals().alloc_bytes.fetch_add(bytes) + bytes;
    std::fprintf(stderr,
                 "[ZEROMEM_CENSUS] alloc bytes=%zu total_count=%zu total_bytes=%zu (%.2f MiB)\n",
                 bytes,
                 n,
                 b,
                 static_cast<double>(b) / (1024.0 * 1024.0));
    std::fflush(stderr);
}

void census_import(size_t bytes) {
    if (!census_on()) {
        return;
    }
    const auto n = totals().import_count.fetch_add(1) + 1;
    const auto b = totals().import_bytes.fetch_add(bytes) + bytes;
    std::fprintf(stderr,
                 "[ZEROMEM_CENSUS] import bytes=%zu total_count=%zu total_bytes=%zu (%.2f MiB)\n",
                 bytes,
                 n,
                 b,
                 static_cast<double>(b) / (1024.0 * 1024.0));
    std::fflush(stderr);
}

void census_refused(size_t bytes, const char* reason) {
    if (!census_on()) {
        return;
    }
    const auto n = totals().refused_count.fetch_add(1) + 1;
    const auto b = totals().refused_bytes.fetch_add(bytes) + bytes;
    std::fprintf(stderr,
                 "[ZEROMEM_CENSUS] refused bytes=%zu reason=%s total_count=%zu total_bytes=%zu (%.2f MiB)\n",
                 bytes,
                 reason,
                 n,
                 b,
                 static_cast<double>(b) / (1024.0 * 1024.0));
    std::fflush(stderr);
}
}  // anonymous namespace

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const size_t bytes,
                 const size_t alignment,
                 const bool is_input)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()),
      _size(bytes == 0 ? alignment : (bytes + alignment - 1) & ~(alignment - 1)) {
    uint32_t zero_memory_flag = 0;
    if (is_input) {
        zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    }

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, zero_memory_flag};
    THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocHost",
                                zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr));

    _id = zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _ptr);
    OPENVINO_ASSERT(_id != 0, "Failed to get memory allocation id of the allocated memory");

    census_alloc(_size);
}

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const void* data,
                 const size_t bytes,
                 const bool is_input,
                 const bool standard_allocation)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()),
      _size(bytes) {
    if (standard_allocation) {
        if (!_init_structs->isExternalMemoryStandardAllocationSupported()) {
            census_refused(_size, "driver_no_standard_allocation_support");
            throw ZeroMemException("Importing standard allocation is not supported with this driver version");
        }

        if (!utils::memory_and_size_aligned_to_standard_page_size(data, _size)) {
            census_refused(_size, "not_page_aligned");
            throw ZeroMemException(
                "Importing standard allocation is not supported if memory is not aligned to standard page size");
        }

        // Reject the import only when the region genuinely overlaps a previously imported
        // allocation. Probe the last valid byte (data + _size - 1) so that a buffer whose end
        // merely abuts an adjacent allocation is still importable. Other cases are handled by
        // the driver.
        if (_size > 0 && zeroUtils::get_l0_context_memory_allocation_id(
                             _init_structs->getContext(),
                             static_cast<void*>(static_cast<uint8_t*>(const_cast<void*>(data)) + _size - 1)) > 0) {
            census_refused(_size, "part_of_existing_allocation");
            throw ZeroMemException("Can not import a memory which is part of an existing allocation");
        }

        uint32_t zero_memory_flag = 0;
        if (is_input) {
            zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
        }
        ze_external_memmap_sysmem_ext_desc_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC,
                                                              nullptr,
                                                              const_cast<void*>(data),
                                                              _size};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, zero_memory_flag};
        auto result = zeMemAllocHost(_init_structs->getContext(), &desc, _size, utils::STANDARD_PAGE_SIZE, &_ptr);

        if (result != ZE_RESULT_SUCCESS) {
            census_refused(_size, ze_result_to_string(result).c_str());
            throw ZeroMemException("Importing memory failed with result " + ze_result_to_string(result) + " - " +
                                   ze_result_to_description(result).c_str());
        }
    } else {
        OPENVINO_ASSERT(_init_structs->isExternalMemoryFdWin32Supported(),
                        "Remote tensor functionality is not supported with this driver version");

        OPENVINO_ASSERT(data != nullptr, "Data pointer for importing memory can't be null");
#ifdef _WIN32
        ze_external_memory_import_win32_handle_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
                                                                  nullptr,
                                                                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                                                                  const_cast<void*>(data),
                                                                  nullptr};
#else
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(data))};
#endif
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, 0};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, _size, utils::STANDARD_PAGE_SIZE, &_ptr));
    }

    _id = zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _ptr);
    OPENVINO_ASSERT(_id != 0, "Failed to get memory allocation id of the imported memory");

    census_import(_size);
}

void* ZeroMem::data() {
    return _ptr;
}

size_t ZeroMem::size() {
    return _size;
}

uint64_t ZeroMem::id() {
    return _id;
}

ZeroMem::~ZeroMem() {
    auto ze_context = _init_structs->getContext();
    if (ze_context == nullptr) {
        _logger.warning("Context is null while trying to free memory with id %llu. Memory might be already freed.",
                        _id);
        return;
    }

    auto result = zeMemFree(ze_context, _ptr);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("L0 zeMemFree result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
    }
}

}  // namespace intel_npu
