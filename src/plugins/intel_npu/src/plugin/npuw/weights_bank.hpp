// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "lazy_tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "orc.hpp"
#include "orc/schema_npuw.hpp"

namespace ov {
namespace npuw {
// Forward declaration
class LLMCompiledModel;
class CompiledModel;
namespace weights {

class Bank {
public:
    static constexpr ov::npuw::orc::TypeId kOrcType =
        static_cast<ov::npuw::orc::TypeId>(ov::npuw::orc::schema_npuw::WeightsBank::ID);
    // Version 0 is the frozen baseline on the wire. Any further layout changes
    // must be introduced through a new versioned payload rather than by mutating v0.
    static constexpr ov::npuw::orc::Version kOrcVersion = 0u;

    Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name);

    // Register LazyTensor in a bank if it's not there. Returns LazyTensor's unique id
    int64_t registerLT(const LazyTensor& tensor, const std::string& device);

    // Get registered, allocated and evaluated tensor on a specified device
    ov::Tensor get(int64_t uid, const std::string& device);

    // Evaluate and allocate all LazyTensors in the bank
    void evaluate_and_allocate();

    bool is_remote(int64_t uid) const;

    std::string get_name() const;

    // XPU shared buffer support: after evaluate_and_allocate(), consolidate
    // transformed weights into the XPU shared buffer region [alloc_offset, size).
    void set_xpu_shared_buffer(void* ptr, size_t size, size_t alloc_offset);
    void consolidate_to_xpu_buffer();
    size_t get_total_tensor_bytes() const;

    // XPU raw weight buffers (the per-weight GPU-shared buffers). Plain un-transformed
    // Const closures whose source pointer lies in any of these ranges are byte-identical
    // to a shared buffer, so they are pointed AT it (zero-copy) instead of being copied
    // into the persistent buffer — giving GPU and NPU one physical copy. The argument is
    // the serialized "ptr:size;ptr:size;..." range list.
    void set_xpu_raw_ranges(const std::string& serialized);

private:
    friend class ov::npuw::LLMCompiledModel;
    friend class ov::npuw::CompiledModel;
    friend void ov::npuw::orc::serialize(ov::npuw::orc::Stream& stream, ov::npuw::weights::Bank& var);

    struct StoredTensor {
        LazyTensor lt;
        ov::Tensor tensor;
    };
    // Bank for specified device and their allocated memory
    struct DeviceBank {
        std::unordered_map<int64_t, StoredTensor> storage;
        std::unordered_map<LazyTensor, int64_t, LazyTensor::Hash> registered_tensors;
    };
    std::unordered_map<std::string, DeviceBank> m_device_banks;

    void evaluate_cpu(DeviceBank& device_bank, const std::vector<LazyTensor>& to_process);
    void evaluate_and_allocate_on_device(DeviceBank& device_bank,
                                         const std::vector<LazyTensor>& to_process,
                                         const std::string& device);

    void serialize(ov::npuw::orc::Stream& stream);
    void read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device);

    mutable std::mutex m_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::string m_alloc_device;
    int64_t uid_count = 0;
    std::string m_bank_name;

    // XPU shared buffer for weight consolidation
    void* m_xpu_shared_ptr = nullptr;
    size_t m_xpu_shared_size = 0;
    size_t m_xpu_alloc_offset = 0;

    // XPU raw weight buffers (shared with the GPU side) for zero-copy Const closures.
    std::vector<std::pair<const void*, size_t>> m_xpu_raw_ranges;

    // Returns the raw-buffer pointer for a closure that is a plain Const resident in
    // the raw buffer, or nullptr if it must be copied into the persistent buffer.
    void* raw_resident_ptr(const StoredTensor& stored) const;
};

std::shared_ptr<Bank> bank(const std::string& bank_name,
                           const std::shared_ptr<const ov::ICore>& core,
                           const std::string& alloc_device);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
