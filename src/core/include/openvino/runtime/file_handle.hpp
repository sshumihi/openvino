// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <limits>

namespace ov {
#ifdef _WIN32
// Windows uses HANDLE (void*) for file handles
using FileHandle = void*;
#else
// Linux/Unix uses int for file descriptors
using FileHandle = int;
#endif

/**
 * @brief Type definition for file handle provider callback (cross-platform).
 * Function that takes no arguments and returns a platform-specific file handle.
 * The callback implementation must release ownership, caller should close the FileHandle.
 * On Linux/Unix: returns int (file descriptor)
 * On Windows: returns void* (HANDLE cast to void*)
 * This is useful for scenarios where file access needs to be controlled externally,
 * such as Android content providers or Windows restricted file access scenarios.
 * @ingroup ov_runtime_cpp_api
 */
using FileHandleProvider = std::function<FileHandle()>;

/**
 * @brief A file handle together with the byte range inside it that holds the data of interest.
 *
 * Use this when the payload is embedded in a larger file rather than being the whole file, so the
 * consumer must map a window instead of the entire object. The offset needs no page alignment: the
 * mapping helpers align it internally and return a pointer to the requested byte.
 *
 * @c offset and @c size are properties of the mapping the provider hands out, so they belong with
 * the handle. Passing them separately lets the two drift apart when a provider is invoked more than
 * once.
 * @ingroup ov_runtime_cpp_api
 */
struct FileRegion {
    /// @brief The whole object behind @p file_handle.
    FileRegion(FileHandle file_handle) : handle(file_handle) {}

    /// @brief The @p region_size bytes at @p region_offset of the object behind @p file_handle.
    FileRegion(FileHandle file_handle, std::size_t region_offset, std::size_t region_size)
        : handle(file_handle),
          offset(region_offset),
          size(region_size) {}

    FileHandle handle;
    std::size_t offset = 0;
    /// @brief Byte length of the region. The default maps from @c offset to the end of the object.
    std::size_t size = std::numeric_limits<std::size_t>::max();
};

/**
 * @brief Type definition for file region provider callback (cross-platform).
 *
 * Same ownership contract as @ref FileHandleProvider: the callback releases ownership of the handle
 * it returns, and the caller closes it. A provider may be called more than once, and each call must
 * yield a handle the caller may close independently.
 *
 * The single-argument @ref FileRegion constructor is deliberately implicit, so a callback that
 * returns a bare handle still satisfies this type and maps the whole object.
 * @ingroup ov_runtime_cpp_api
 */
using FileRegionProvider = std::function<FileRegion()>;
}  // namespace ov
