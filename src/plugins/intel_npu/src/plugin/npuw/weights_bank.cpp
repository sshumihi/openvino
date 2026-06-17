// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"
#include "xpu_debug.hpp"

#include <iostream>
#include <map>

#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "serialization.hpp"
#include "util.hpp"

namespace {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace

using ov::npuw::weights::Bank;
using ov::npuw::weights::LazyTensor;

class BankManager {
public:
    static BankManager& getInstance() {
        static BankManager instance;
        return instance;
    }

private:
    BankManager() {}
    BankManager(const BankManager&) = delete;
    void operator=(const BankManager&) = delete;

public:
    // Public API
    std::shared_ptr<Bank> getBank(const std::string& bank_name,
                                  const std::shared_ptr<const ov::ICore>& core,
                                  const std::string& alloc_device);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

Bank::Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name)
    : m_core(core),
      m_alloc_device(alloc_device),
      m_bank_name(bank_name) {
    if (m_bank_name.empty()) {
        auto unique_name = ov::npuw::util::generate_random_string();
        LOG_WARN("Got an empty name for weights bank! Using a uniquely generated instead: " << unique_name);
        m_bank_name = unique_name;
    }
}

int64_t Bank::registerLT(const LazyTensor& tensor, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks[device_for_alloc];

    auto iter_registered = device_bank.registered_tensors.find(tensor);
    if (iter_registered == device_bank.registered_tensors.end()) {
        auto uid = uid_count++;
        device_bank.registered_tensors[tensor] = uid;
        device_bank.storage[uid] = {tensor, ov::Tensor()};
        return uid;
    } else {
        // Already registered - can be safely detach the incoming tensor
        const_cast<LazyTensor&>(tensor).detach();
    }

    return iter_registered->second;
}

ov::Tensor Bank::get(int64_t uid, const std::string& device) {
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks.at(device_for_alloc);
    auto iter_device = device_bank.storage.find(uid);

    NPUW_ASSERT(iter_device != device_bank.storage.end() && iter_device->second.tensor &&
                "Tensor should be registered and allocated first!");

    return iter_device->second.tensor;
}

struct TensorToAllocate {
    LazyTensor::Meta meta;
    ov::Tensor allocated_tensor;
    int64_t uid;
};

void Bank::evaluate_and_allocate() {
    std::unique_lock guard(m_mutex);

    for (auto&& bank : m_device_banks) {
        const auto& device_for_alloc = bank.first;
        auto& device_bank = bank.second;

        std::vector<LazyTensor> to_process;
        to_process.reserve(device_bank.storage.size());
        for (const auto& el : device_bank.storage) {
            // Add non-allocated tensors for furter evaluation and allocation
            if (!el.second.tensor) {
                to_process.push_back(el.second.lt);
            }
        }

        if (device_for_alloc == "CPU") {
            evaluate_cpu(device_bank, to_process);
        } else {
            evaluate_and_allocate_on_device(device_bank, to_process, device_for_alloc);
        }
    }  // for (m_device_banks)

    // Post-evaluation consolidation: copy transformed weights into XPU shared buffer
    if (m_xpu_shared_ptr) {
        consolidate_to_xpu_buffer();
    }
}

void Bank::evaluate_cpu(Bank::DeviceBank& device_bank, const std::vector<LazyTensor>& to_process) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    ov::parallel_for(to_process.size(), [&](std::size_t idx) {
        const auto& lt = to_process[idx];
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        auto uid = iter_device_registered->second;

        // XPU shared buffer: if this is a plain un-transformed Const whose source data lies fully
        // inside a registered raw buffer (the GPU+NPU shared malloc), store a zero-copy VIEW of it
        // instead of eval + allocate + copy. This avoids a full host MATERIALIZATION of the weights
        // at compile time (the closure already lives in the shared buffer the NPU reads at inference).
        if (!m_xpu_raw_ranges.empty()) {
            auto cs = lt.const_source();  // {ptr, byte_size} for a plain Const, else {nullptr, 0}
            if (cs.first) {
                const auto* p = static_cast<const uint8_t*>(cs.first);
                for (const auto& [raw, size] : m_xpu_raw_ranges) {
                    const auto* rp = static_cast<const uint8_t*>(raw);
                    if (p >= rp && p + cs.second <= rp + size) {
                        auto meta = lt.eval_meta();
                        device_bank.storage.at(uid).tensor =
                            ov::Tensor(meta.type, meta.shape, const_cast<void*>(cs.first));
                        const_cast<LazyTensor&>(lt).detach();
                        return;  // zero-copy view; skip materialization
                    }
                }
            }
        }

        auto t = lt.eval();
        device_bank.storage.at(uid).tensor = ov::Tensor(t.get_element_type(), t.get_shape());
        // Get ownership of the weights, might be a mmaped object during import
        t.copy_to(device_bank.storage.at(uid).tensor);
        const_cast<LazyTensor&>(lt).detach();
    });
}

// Note: there are no locks in this function's parallel_for
// At this point all the LazyTensor->Tensor pairs in the map are already
// allocated and there are no conflicting reads/writes since all the tensors are unique.

// FIXME: this whole flow could be improved for the same bank
// processing from different threads. We could separate LazyTensors
// evaluation and the bank access. But it requires additional rework.
void Bank::evaluate_and_allocate_on_device(Bank::DeviceBank& device_bank,
                                           const std::vector<LazyTensor>& to_process,
                                           const std::string& device) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    std::vector<TensorToAllocate> uids_to_allocated;
    uids_to_allocated.reserve(uid_count);
    std::size_t raw_views = 0;

    for (const auto& lt : to_process) {
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        auto uid = iter_device_registered->second;

        // XPU shared buffer: if this is a plain un-transformed Const whose source lies fully inside a
        // registered raw buffer (the GPU+NPU shared malloc), store a zero-copy VIEW of it instead of
        // allocating an NPU host tensor and copying the weight in. The NPU imports this same shared
        // buffer at inference, so the device copy is pure duplication. (consolidate_to_xpu_buffer
        // does the same post-eval; doing it here avoids materializing the weights at all.)
        if (!m_xpu_raw_ranges.empty()) {
            auto cs = lt.const_source();  // {ptr, byte_size} for a plain Const, else {nullptr, 0}
            if (cs.first) {
                const auto* p = static_cast<const uint8_t*>(cs.first);
                bool resident = false;
                for (const auto& [raw, size] : m_xpu_raw_ranges) {
                    const auto* rp = static_cast<const uint8_t*>(raw);
                    if (p >= rp && p + cs.second <= rp + size) {
                        resident = true;
                        break;
                    }
                }
                if (resident) {
                    auto meta = lt.eval_meta();
                    device_bank.storage.at(uid).tensor =
                        ov::Tensor(meta.type, meta.shape, const_cast<void*>(cs.first));
                    const_cast<LazyTensor&>(lt).detach();
                    ++raw_views;
                    continue;  // skip device allocation + materialization
                }
            }
        }

        uids_to_allocated.push_back({lt.eval_meta(), ov::Tensor(), uid});
    }
    if (std::getenv("XPU_MEM_DEBUG"))
        std::cerr << "[DIAG] eval_on_device(" << device << "): to_process=" << to_process.size()
                  << " ranges=" << m_xpu_raw_ranges.size() << " raw_views=" << raw_views
                  << " materialized=" << uids_to_allocated.size() << std::endl;
    // Sort by UIDs, lowest first
    std::sort(uids_to_allocated.begin(),
              uids_to_allocated.end(),
              [](const TensorToAllocate& a, const TensorToAllocate& b) {
                  return a.uid < b.uid;
              });

    // Allocate memory sequentially - in order of UID
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    for (auto&& allocated : uids_to_allocated) {
        ov::SoPtr<ov::ITensor> remote_tensor =
            remote_ctx->create_host_tensor(allocated.meta.type, allocated.meta.shape);
        allocated = {allocated.meta, ov::make_tensor(remote_tensor), allocated.uid};
    }

    // Evaluate and copy into the device memory
    ov::parallel_for(uids_to_allocated.size(), [&](std::size_t idx) {
        auto& allocated = uids_to_allocated[idx];
        auto& stored_tensor = device_bank.storage.at(allocated.uid);

        auto transformed = stored_tensor.lt.eval();
        transformed.copy_to(allocated.allocated_tensor);
        stored_tensor.tensor = std::move(allocated.allocated_tensor);

        // Detach the evaluated LazyTensor from its memory here - when it is 100%
        // not needed anymore (transformations, if any, and copies are done)
        // Note: this is the non-CPU path!
        const_cast<LazyTensor&>(stored_tensor.lt).detach();
    });
}

void Bank::set_xpu_shared_buffer(void* ptr, size_t size, size_t alloc_offset) {
    m_xpu_shared_ptr = ptr;
    m_xpu_shared_size = size;
    m_xpu_alloc_offset = alloc_offset;
    LOG_INFO("XPU shared buffer set: ptr=" << ptr << " size=" << size << " alloc_offset=" << alloc_offset);
}

void Bank::set_xpu_raw_ranges(const std::string& serialized) {
    // Parse "ptr:size;ptr:size;..." — one entry per GPU-shared per-weight buffer.
    m_xpu_raw_ranges.clear();
    std::size_t pos = 0;
    while (pos < serialized.size()) {
        auto semi = serialized.find(';', pos);
        auto entry = serialized.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        auto colon = entry.find(':');
        if (colon != std::string::npos) {
            auto ptr = reinterpret_cast<const void*>(std::stoull(entry.substr(0, colon)));
            auto size = static_cast<std::size_t>(std::stoull(entry.substr(colon + 1)));
            m_xpu_raw_ranges.emplace_back(ptr, size);
        }
        if (semi == std::string::npos)
            break;
        pos = semi + 1;
    }
    LOG_INFO("XPU raw ranges set: " << m_xpu_raw_ranges.size() << " buffers");
}

void* Bank::raw_resident_ptr(const StoredTensor& stored) const {
    // Only when raw buffers are registered and the closure is a plain un-transformed
    // Const whose original data pointer + size lies entirely within one of them.
    if (m_xpu_raw_ranges.empty() || !stored.tensor) {
        return nullptr;
    }
    auto src = stored.lt.const_source();  // {ptr, byte_size}
    if (!src.first) {
        return nullptr;
    }
    // The evaluated tensor's bytes must match the const source (no element-type change).
    if (src.second != stored.tensor.get_byte_size()) {
        return nullptr;
    }
    auto* p = static_cast<const uint8_t*>(src.first);
    for (const auto& [raw, size] : m_xpu_raw_ranges) {
        auto* rp = static_cast<const uint8_t*>(raw);
        if (p >= rp && p + src.second <= rp + size) {
            return const_cast<void*>(src.first);
        }
    }
    return nullptr;
}

void Bank::consolidate_to_xpu_buffer() {
    // Note: when called from evaluate_and_allocate(), m_mutex is already held.
    // When called from LLMCompiledModel::set_property() (deferred path),
    // evaluations are complete and no concurrent access occurs.
    auto* buf = static_cast<uint8_t*>(m_xpu_shared_ptr);
    auto* buf_end = buf + m_xpu_shared_size;
    size_t offset = m_xpu_alloc_offset;
    size_t relocated_count = 0;
    size_t relocated_bytes = 0;
    size_t skipped_count = 0;
    size_t raw_shared_count = 0;
    size_t raw_shared_bytes = 0;

    // Per-type statistics: element_type_name -> {count, total_bytes}
    std::map<std::string, std::pair<size_t, size_t>> type_stats;
    // Per-transform statistics: transform_name -> {count, total_bytes}
    std::map<std::string, std::pair<size_t, size_t>> transform_stats;

    for (auto& [device, device_bank] : m_device_banks) {
        ::ov::npuw::xpu_dbg() << "[DIAG] consolidate_to_xpu_buffer: device=" << device
                  << " storage_size=" << device_bank.storage.size() << std::endl;
        for (auto& [uid, stored] : device_bank.storage) {
            if (!stored.tensor) {
                continue;
            }

            auto* tensor_data = static_cast<uint8_t*>(stored.tensor.data());
            size_t tensor_size = stored.tensor.get_byte_size();

            // Determine LazyTensor transform type
            std::string transform_name = "none";
            auto transforms = stored.lt.get_transformations();
            if (!transforms.empty()) {
                std::visit(overloaded{
                    [&](const ov::npuw::weights::op::Const&)   { transform_name = "Const"; },
                    [&](const ov::npuw::weights::op::Unpack&)  { transform_name = "Unpack"; },
                    [&](const ov::npuw::weights::op::Permute&) { transform_name = "Permute"; },
                    [&](const ov::npuw::weights::op::Convert&) { transform_name = "Convert"; },
                    [&](const ov::npuw::weights::op::Concat&)  { transform_name = "Concat"; },
                    [&](const ov::npuw::weights::op::Gather&)  { transform_name = "Gather"; },
                }, transforms[0]);
                if (transforms.size() > 1) {
                    transform_name += "+" + std::to_string(transforms.size() - 1) + "more";
                }
            }

            auto type_name = stored.tensor.get_element_type().get_type_name();
            type_stats[type_name].first++;
            type_stats[type_name].second += tensor_size;
            transform_stats[transform_name].first++;
            transform_stats[transform_name].second += tensor_size;

            ::ov::npuw::xpu_dbg() << "[DIAG]   [uid=" << uid << "] transform=" << transform_name
                      << " type=" << stored.tensor.get_element_type()
                      << " shape=" << stored.tensor.get_shape()
                      << " bytes=" << tensor_size << std::endl;

            // Skip tensors already in the shared buffer range
            if (tensor_data >= buf && tensor_data + tensor_size <= buf_end) {
                skipped_count++;
                continue;
            }

            // Zero-copy raw-buffer sharing: a plain Const closure that is byte-identical
            // to the GPU-shared raw buffer is pointed AT the raw buffer instead of being
            // copied into the persistent buffer, so GPU and NPU hold one physical copy.
            if (void* raw_ptr = raw_resident_ptr(stored)) {
                stored.tensor = ov::Tensor(stored.tensor.get_element_type(), stored.tensor.get_shape(), raw_ptr);
                raw_shared_count++;
                raw_shared_bytes += tensor_size;
                continue;
            }

            // Buffer must be exactly sized — overflow is a bug
            NPUW_ASSERT(offset + tensor_size <= m_xpu_shared_size &&
                        "XPU shared buffer overflow: buffer was not sized correctly");

            // Copy tensor data to shared buffer and replace the tensor
            std::memcpy(buf + offset, tensor_data, tensor_size);
            stored.tensor = ov::Tensor(stored.tensor.get_element_type(), stored.tensor.get_shape(), buf + offset);
            relocated_count++;
            relocated_bytes += tensor_size;
            offset += tensor_size;
        }
    }

    ::ov::npuw::xpu_dbg() << "[DIAG] consolidate_to_xpu_buffer: relocated " << relocated_count << " tensors (" << relocated_bytes
              << " bytes), skipped " << skipped_count
              << ", raw-shared " << raw_shared_count << " tensors (" << raw_shared_bytes
              << " bytes, zero-copy from GPU raw buffer)"
              << ", final offset=" << offset << "/" << m_xpu_shared_size << std::endl;

    ::ov::npuw::xpu_dbg() << "[DIAG] === Summary by element type ===" << std::endl;
    for (const auto& [type, stats] : type_stats) {
        ::ov::npuw::xpu_dbg() << "[DIAG]   " << type << ": " << stats.first << " tensors, "
                  << (stats.second / 1048576.0) << " MB" << std::endl;
    }
    ::ov::npuw::xpu_dbg() << "[DIAG] === Summary by transform type ===" << std::endl;
    for (const auto& [tform, stats] : transform_stats) {
        ::ov::npuw::xpu_dbg() << "[DIAG]   " << tform << ": " << stats.first << " tensors, "
                  << (stats.second / 1048576.0) << " MB" << std::endl;
    }
}

size_t Bank::get_total_tensor_bytes() const {
    std::unique_lock guard(m_mutex);
    size_t total = 0;
    for (const auto& [device, device_bank] : m_device_banks) {
        for (const auto& [uid, stored] : device_bank.storage) {
            if (stored.tensor) {
                // Raw-resident Const closures are not copied into the persistent buffer
                // (they are pointed at the shared raw buffer), so exclude them from the
                // persistent-buffer sizing.
                if (raw_resident_ptr(stored)) {
                    continue;
                }
                total += stored.tensor.get_byte_size();
            }
        }
    }
    return total;
}

bool Bank::is_remote(int64_t uid) const {
    // FIXME: make generic
    std::unique_lock guard(m_mutex);

    auto npu_bank = m_device_banks.find("NPU");
    if (npu_bank != m_device_banks.end()) {
        if (npu_bank->second.storage.find(uid) != npu_bank->second.storage.end()) {
            // Found in NPU bank so considered remote (utterly wrong for the generic case)
            return true;
        }
    }
    return false;
}

void Bank::serialize(ov::npuw::orc::Stream& stream) {
    LOG_INFO("Serializing weights bank...");
    LOG_BLOCK();

    std::unique_lock guard(m_mutex);

    std::size_t bank_size = m_device_banks.size();
    stream & bank_size;

    for (const auto& elem : m_device_banks) {
        const auto& device = elem.first;
        const auto& device_bank = elem.second;
        auto storage_size = device_bank.storage.size();
        stream & device & storage_size;
        // Write tensors sequentially according to sorted uids for better memory allocation and utilization
        std::set<int64_t> uids;
        for (const auto& t_pair : device_bank.storage) {
            uids.insert(t_pair.first);
        }

        for (const auto& uid : uids) {
            stream & uid;
            auto tensor = device_bank.storage.at(uid).tensor;
            transfer_tensor(stream, tensor);
        }
    }

    LOG_INFO("DONE.");
}

void Bank::read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device) {
    // This method is supposed to be used only during deserialization
    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks[device];
    auto iter_device = device_bank.storage.find(uid);

    if (iter_device != device_bank.storage.end()) {
        // Shouldn't be possible
        NPUW_ASSERT(false);
        return;
    }

    if (device == "CPU") {
        // Just read deserialized tensor into the bank
        transfer_tensor(stream, device_bank.storage[uid].tensor);
        return;
    }

    // Need to allocate on device and copy deserialized tensor to that memory
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    transfer_tensor(stream,
                    device_bank.storage[uid].tensor,
                    [&remote_ctx](const ov::element::Type& type, const ov::Shape& shape) {
                        ov::SoPtr<ov::ITensor> remote_tensor = remote_ctx->create_host_tensor(type, shape);
                        return ov::make_tensor(remote_tensor);
                    });
    NPUW_ASSERT(device_bank.storage[uid].tensor && "Remote tensor should be initialized during bank deserialize");
    device_bank.storage[uid].lt = LazyTensor();
}

std::string Bank::get_name() const {
    return m_bank_name;
}

void ov::npuw::orc::serialize(Stream& stream, ov::npuw::weights::Bank& var) {
    if (stream.output()) {
        var.serialize(stream);
    } else {
        LOG_INFO("Deserializing weights bank...");
        LOG_BLOCK();

        std::size_t bank_size = 0;
        stream & bank_size;

        for (std::size_t i = 0; i < bank_size; ++i) {
            std::string device;
            stream & device;
            std::size_t storage_size = 0;
            stream & storage_size;
            for (std::size_t j = 0; j < storage_size; ++j) {
                int64_t uid = -1;
                stream & uid;
                var.read_and_add_tensor(stream, uid, device);
            }
        }

        LOG_INFO("DONE.");
    }
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name,
                                           const std::shared_ptr<const ov::ICore>& core,
                                           const std::string& alloc_device) {
    std::unique_lock guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end() || iter->second.expired()) {
        auto bank = std::make_shared<Bank>(core, alloc_device, bank_name);
        m_bank_map[bank_name] = bank;
        return bank;
    }
    return iter->second.lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name,
                                              const std::shared_ptr<const ov::ICore>& core,
                                              const std::string& alloc_device) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>(core, alloc_device, bank_name);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core, alloc_device);
}
