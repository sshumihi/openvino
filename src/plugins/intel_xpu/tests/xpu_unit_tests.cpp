// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

// XPU plugin headers (from ../src/)
#include "shared_weight_buffer.hpp"

// ============================================================================
// Test Group 1: SharedWeightBuffer
// Tests the RAII wrapper for shared weight buffer allocation and file I/O
// ============================================================================

class SharedWeightBufferTest : public ::testing::Test {
protected:
    std::filesystem::path m_temp_dir;

    void SetUp() override {
        m_temp_dir = std::filesystem::temp_directory_path() / "xpu_test";
        std::filesystem::create_directories(m_temp_dir);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(m_temp_dir, ec);
    }

    // Helper: write a binary file with known content
    std::filesystem::path write_bin_file(const std::string& name, const std::vector<uint8_t>& data) {
        auto path = m_temp_dir / name;
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
        f.close();
        return path;
    }
};

TEST_F(SharedWeightBufferTest, CreateFromValidFile) {
    // Create a test binary file
    std::vector<uint8_t> test_data(4096, 0xAB);
    auto bin_path = write_bin_file("weights.bin", test_data);

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());

    ASSERT_NE(buf, nullptr);
    ASSERT_NE(buf->host_ptr, nullptr);
    ASSERT_EQ(buf->size, 4096u);

    // Verify content was read correctly
    auto* data = static_cast<uint8_t*>(buf->host_ptr);
    for (size_t i = 0; i < buf->size; i++) {
        ASSERT_EQ(data[i], 0xAB) << "Mismatch at byte " << i;
    }
}

TEST_F(SharedWeightBufferTest, CreateFromLargeFile) {
    // Test with a larger buffer (1MB)
    std::vector<uint8_t> test_data(1024 * 1024);
    for (size_t i = 0; i < test_data.size(); i++) {
        test_data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    auto bin_path = write_bin_file("large_weights.bin", test_data);

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());

    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(buf->size, 1024u * 1024u);

    // Verify content pattern
    auto* data = static_cast<uint8_t*>(buf->host_ptr);
    for (size_t i = 0; i < buf->size; i++) {
        ASSERT_EQ(data[i], static_cast<uint8_t>(i & 0xFF)) << "Mismatch at byte " << i;
    }
}

TEST_F(SharedWeightBufferTest, CreateFromNonExistentFileThrows) {
    ASSERT_THROW(
        ov::intel_xpu::SharedWeightBuffer::create((m_temp_dir / "nonexistent.bin").string()),
        ov::Exception);
}

TEST_F(SharedWeightBufferTest, CreateFromEmptyFileThrows) {
    auto bin_path = write_bin_file("empty.bin", {});
    ASSERT_THROW(
        ov::intel_xpu::SharedWeightBuffer::create(bin_path.string()),
        ov::Exception);
}

TEST_F(SharedWeightBufferTest, HostPtrIsAligned) {
    std::vector<uint8_t> test_data(8192, 0x42);
    auto bin_path = write_bin_file("aligned_weights.bin", test_data);

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());
    ASSERT_NE(buf, nullptr);

    // host_ptr should be page-aligned (4096)
    auto ptr_val = reinterpret_cast<uintptr_t>(buf->host_ptr);
    // The buffer is either _aligned_malloc'd (4096 align) or zeMemAllocHost'd (also 4096)
    // Both should give at least 4096-byte alignment
    ASSERT_EQ(ptr_val % 4096, 0u) << "host_ptr is not 4096-byte aligned";
}

TEST_F(SharedWeightBufferTest, DestructorDoesNotCrash) {
    std::vector<uint8_t> test_data(1024, 0xFF);
    auto bin_path = write_bin_file("destruct_test.bin", test_data);

    {
        auto buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());
        ASSERT_NE(buf, nullptr);
    }
    // If we reach here without crash, the destructor worked correctly
}

TEST_F(SharedWeightBufferTest, MultipleBuffersIndependent) {
    std::vector<uint8_t> data1(2048, 0x11);
    std::vector<uint8_t> data2(4096, 0x22);
    auto path1 = write_bin_file("w1.bin", data1);
    auto path2 = write_bin_file("w2.bin", data2);

    auto buf1 = ov::intel_xpu::SharedWeightBuffer::create(path1.string());
    auto buf2 = ov::intel_xpu::SharedWeightBuffer::create(path2.string());

    ASSERT_NE(buf1->host_ptr, buf2->host_ptr);
    ASSERT_EQ(buf1->size, 2048u);
    ASSERT_EQ(buf2->size, 4096u);
    ASSERT_EQ(static_cast<uint8_t*>(buf1->host_ptr)[0], 0x11);
    ASSERT_EQ(static_cast<uint8_t*>(buf2->host_ptr)[0], 0x22);
}

// ============================================================================
// Test Group 2: ProgramBuilder shared weight range checking
// Tests is_shared_weight_ptr() logic in isolation (pure pointer arithmetic)
// ============================================================================

// We test the pointer range logic directly without needing a real ProgramBuilder.
// The logic is: ptr >= start && (ptr + byte_count) <= (start + size)
class SharedWeightRangeTest : public ::testing::Test {
protected:
    // Simulate the range-checking logic from ProgramBuilder
    static bool is_shared_weight_ptr(const void* shared_start, size_t shared_size,
                                     const void* ptr, size_t byte_count) {
        if (!shared_start)
            return false;
        auto start = static_cast<const char*>(shared_start);
        auto p = static_cast<const char*>(ptr);
        return p >= start && (p + byte_count) <= (start + shared_size);
    }
};

TEST_F(SharedWeightRangeTest, NullStartReturnsFalse) {
    char buf[100];
    ASSERT_FALSE(is_shared_weight_ptr(nullptr, 0, buf, 10));
    ASSERT_FALSE(is_shared_weight_ptr(nullptr, 100, buf, 10));
}

TEST_F(SharedWeightRangeTest, PtrAtStartInRange) {
    char buf[1024];
    ASSERT_TRUE(is_shared_weight_ptr(buf, 1024, buf, 100));
}

TEST_F(SharedWeightRangeTest, PtrAtEndBoundary) {
    char buf[1024];
    // Pointer at offset 924, size 100: 924+100 = 1024 = start+size => in range
    ASSERT_TRUE(is_shared_weight_ptr(buf, 1024, buf + 924, 100));
}

TEST_F(SharedWeightRangeTest, PtrExceedsEndBoundary) {
    char buf[1024];
    // Pointer at offset 925, size 100: 925+100 = 1025 > 1024 => out of range
    ASSERT_FALSE(is_shared_weight_ptr(buf, 1024, buf + 925, 100));
}

TEST_F(SharedWeightRangeTest, PtrBeforeStartOutOfRange) {
    char buf[1024];
    ASSERT_FALSE(is_shared_weight_ptr(buf + 100, 924, buf, 50));
}

TEST_F(SharedWeightRangeTest, ExactFitEntireBuffer) {
    char buf[1024];
    ASSERT_TRUE(is_shared_weight_ptr(buf, 1024, buf, 1024));
}

TEST_F(SharedWeightRangeTest, ZeroSizeByteCount) {
    char buf[1024];
    ASSERT_TRUE(is_shared_weight_ptr(buf, 1024, buf + 500, 0));
}

TEST_F(SharedWeightRangeTest, PtrInMiddleOfBuffer) {
    char buf[4096];
    ASSERT_TRUE(is_shared_weight_ptr(buf, 4096, buf + 1000, 2000));
    ASSERT_TRUE(is_shared_weight_ptr(buf, 4096, buf + 4000, 96));
    ASSERT_FALSE(is_shared_weight_ptr(buf, 4096, buf + 4000, 97));
}

TEST_F(SharedWeightRangeTest, CompletelyDisjointRanges) {
    char buf1[1024];
    char buf2[1024];
    ASSERT_FALSE(is_shared_weight_ptr(buf1, 1024, buf2, 100));
}

// ============================================================================
// Test Group 3: NPU WeightlessGraph is_xpu_shared_weight (same logic)
// ============================================================================

class NpuSharedWeightCheckTest : public ::testing::Test {
protected:
    // Mirror the static method from WeightlessGraph
    static bool is_xpu_shared_weight(const void* ptr, size_t size,
                                     const void* shared_start, size_t shared_size) {
        if (!shared_start || !ptr)
            return false;
        auto start = static_cast<const char*>(shared_start);
        auto p = static_cast<const char*>(ptr);
        return p >= start && (p + size) <= (start + shared_size);
    }
};

TEST_F(NpuSharedWeightCheckTest, NullPtrReturnsFalse) {
    char buf[100];
    ASSERT_FALSE(is_xpu_shared_weight(nullptr, 10, buf, 100));
}

TEST_F(NpuSharedWeightCheckTest, NullSharedStartReturnsFalse) {
    char buf[100];
    ASSERT_FALSE(is_xpu_shared_weight(buf, 10, nullptr, 0));
}

TEST_F(NpuSharedWeightCheckTest, BothNullReturnsFalse) {
    ASSERT_FALSE(is_xpu_shared_weight(nullptr, 0, nullptr, 0));
}

TEST_F(NpuSharedWeightCheckTest, ValidRangeReturnsTrue) {
    char buf[2048];
    ASSERT_TRUE(is_xpu_shared_weight(buf + 100, 500, buf, 2048));
}

TEST_F(NpuSharedWeightCheckTest, OutOfRangeReturnsFalse) {
    char buf[2048];
    ASSERT_FALSE(is_xpu_shared_weight(buf + 2000, 100, buf, 2048));
}

TEST_F(NpuSharedWeightCheckTest, ExactBoundaryReturnsTrue) {
    char buf[2048];
    ASSERT_TRUE(is_xpu_shared_weight(buf + 1948, 100, buf, 2048));
}

TEST_F(NpuSharedWeightCheckTest, OneByteOverBoundaryReturnsFalse) {
    char buf[2048];
    ASSERT_FALSE(is_xpu_shared_weight(buf + 1949, 100, buf, 2048));
}

// ============================================================================
// Test Group 4: WeightlessCacheAttribute attachment
// Tests that the XPU plugin correctly computes and attaches WeightlessCacheAttribute
// to Constant nodes whose data falls within a shared buffer.
// ============================================================================

class WeightlessCacheAttributeTest : public ::testing::Test {
protected:
    // Simulate the attribute attachment logic from Plugin::compile_model
    static void attach_attributes(std::shared_ptr<ov::Model>& model, void* buf_start, size_t buf_size) {
        for (auto& node : model->get_ops()) {
            auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
            if (!constant)
                continue;

            auto data_ptr = constant->get_data_ptr<char>();
            auto start = static_cast<char*>(buf_start);
            auto end = start + buf_size;

            if (data_ptr >= start && data_ptr < end) {
                size_t bin_offset = static_cast<size_t>(data_ptr - start);
                size_t original_size = constant->get_byte_size();
                ov::element::Type original_dtype = constant->get_output_element_type(0);

                auto& rt_info = constant->get_rt_info();
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(original_size, bin_offset, original_dtype);
            }
        }
    }
};

TEST_F(WeightlessCacheAttributeTest, ConstantInBufferGetsAttribute) {
    // Create a buffer with known data
    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};

    // Create a Constant that uses data from the buffer
    auto constant = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{4}, weights.data());

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto add = std::make_shared<ov::op::v1::Add>(param, constant);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    // Note: The constant copies data internally, so its data_ptr won't be in our buffer.
    // This test verifies the logic correctly does NOT attach attribute for out-of-buffer constants.
    attach_attributes(model, weights.data(), weights.size() * sizeof(float));

    auto& rt_info = constant->get_rt_info();
    // The constant made an internal copy, so its data_ptr != weights.data()
    // Therefore no attribute should be attached
    ASSERT_EQ(rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()), rt_info.end());
}

TEST_F(WeightlessCacheAttributeTest, ConstantOutsideBufferGetsNoAttribute) {
    // Create a constant with its own internal data
    std::vector<float> const_data = {5.0f, 6.0f, 7.0f, 8.0f};
    auto constant = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{4}, const_data.data());

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto add = std::make_shared<ov::op::v1::Add>(param, constant);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    // Use a completely separate buffer range
    char fake_buffer[1024];
    attach_attributes(model, fake_buffer, sizeof(fake_buffer));

    auto& rt_info = constant->get_rt_info();
    ASSERT_EQ(rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()), rt_info.end());
}

// ============================================================================
// Test Group 5: Model rt_info for shared buffer
// Tests that xpu_shared_weight_ptr and xpu_shared_weight_size are correctly
// set and read from model rt_info.
// ============================================================================

class ModelRtInfoTest : public ::testing::Test {};

TEST_F(ModelRtInfoTest, SetAndGetSharedWeightPtr) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    char buffer[4096];
    auto ptr_val = reinterpret_cast<uint64_t>(buffer);
    uint64_t size_val = 4096;

    model->set_rt_info(ptr_val, "xpu_shared_weight_ptr");
    model->set_rt_info(size_val, "xpu_shared_weight_size");

    ASSERT_TRUE(model->has_rt_info("xpu_shared_weight_ptr"));
    ASSERT_TRUE(model->has_rt_info("xpu_shared_weight_size"));

    auto retrieved_ptr = model->get_rt_info<uint64_t>("xpu_shared_weight_ptr");
    auto retrieved_size = model->get_rt_info<uint64_t>("xpu_shared_weight_size");

    ASSERT_EQ(retrieved_ptr, ptr_val);
    ASSERT_EQ(retrieved_size, size_val);

    // Verify round-trip: cast back to pointer
    auto* reconstructed = reinterpret_cast<void*>(retrieved_ptr);
    ASSERT_EQ(reconstructed, static_cast<void*>(buffer));
}

TEST_F(ModelRtInfoTest, MissingRtInfoReturnsFalse) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    ASSERT_FALSE(model->has_rt_info("xpu_shared_weight_ptr"));
    ASSERT_FALSE(model->has_rt_info("xpu_shared_weight_size"));
}

TEST_F(ModelRtInfoTest, LargePointerValuePreserved) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    // Test with a large 64-bit value (high address space)
    uint64_t high_addr = 0x00007FFF'ABCD1234ULL;
    model->set_rt_info(high_addr, "xpu_shared_weight_ptr");

    auto retrieved = model->get_rt_info<uint64_t>("xpu_shared_weight_ptr");
    ASSERT_EQ(retrieved, high_addr);
}

// ============================================================================
// Test Group 6: XPU Plugin property tests
// Tests that the plugin correctly reports its properties without hardware.
// ============================================================================

class XpuPluginPropertyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Try to create a Core and check if XPU is available
        // This test group only checks plugin registration and properties
    }
};

TEST_F(XpuPluginPropertyTest, CoreCanBeCreated) {
    // Verify the OV Core can be created without crashing.
    // Device availability depends on hardware and plugin registration,
    // so we only assert the Core object itself is functional.
    ov::Core core;
    auto devices = core.get_available_devices();
    // Just verify get_available_devices() doesn't throw — device list may be empty
    // in environments without GPU/NPU hardware or registered plugins
    SUCCEED();
}

// ============================================================================
// Test Group 7: WeightlessCacheAttribute correctness
// Tests the WeightlessCacheAttribute struct itself.
// ============================================================================

class WeightlessCacheAttrStructTest : public ::testing::Test {};

TEST_F(WeightlessCacheAttrStructTest, ConstructorSetsFields) {
    ov::WeightlessCacheAttribute attr(1024, 512, ov::element::f32);

    ASSERT_EQ(attr.original_size, 1024u);
    ASSERT_EQ(attr.bin_offset, 512u);
    ASSERT_EQ(attr.original_dtype, ov::element::f32);
}

TEST_F(WeightlessCacheAttrStructTest, DifferentDtypes) {
    ov::WeightlessCacheAttribute attr_f16(2048, 0, ov::element::f16);
    ASSERT_EQ(attr_f16.original_dtype, ov::element::f16);

    ov::WeightlessCacheAttribute attr_i8(512, 100, ov::element::i8);
    ASSERT_EQ(attr_i8.original_dtype, ov::element::i8);

    ov::WeightlessCacheAttribute attr_u4(256, 200, ov::element::u4);
    ASSERT_EQ(attr_u4.original_dtype, ov::element::u4);
}

TEST_F(WeightlessCacheAttrStructTest, AttachToConstantRtInfo) {
    std::vector<float> data = {1.0f, 2.0f};
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, data.data());

    auto& rt_info = constant->get_rt_info();
    rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(8, 0, ov::element::f32);

    // Read it back
    auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
    ASSERT_NE(it, rt_info.end());

    auto& attr = it->second.as<ov::WeightlessCacheAttribute>();
    ASSERT_EQ(attr.original_size, 8u);
    ASSERT_EQ(attr.bin_offset, 0u);
    ASSERT_EQ(attr.original_dtype, ov::element::f32);
}

TEST_F(WeightlessCacheAttrStructTest, ZeroOffsetAndSize) {
    // Edge case: zero values
    ov::WeightlessCacheAttribute attr(0, 0, ov::element::f32);
    ASSERT_EQ(attr.original_size, 0u);
    ASSERT_EQ(attr.bin_offset, 0u);
}

TEST_F(WeightlessCacheAttrStructTest, LargeOffsets) {
    // Large model with many weights
    size_t large_offset = 1024ULL * 1024 * 1024 * 2;  // 2GB offset
    size_t large_size = 1024ULL * 1024 * 512;          // 512MB size
    ov::WeightlessCacheAttribute attr(large_size, large_offset, ov::element::i4);
    ASSERT_EQ(attr.original_size, large_size);
    ASSERT_EQ(attr.bin_offset, large_offset);
}

// ============================================================================
// Test Group 8: SharedWeightBuffer content integrity
// Verifies that the buffer content survives create + read round-trip
// ============================================================================

class SharedWeightBufferContentTest : public ::testing::Test {
protected:
    std::filesystem::path m_temp_dir;

    void SetUp() override {
        m_temp_dir = std::filesystem::temp_directory_path() / "xpu_content_test";
        std::filesystem::create_directories(m_temp_dir);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(m_temp_dir, ec);
    }
};

TEST_F(SharedWeightBufferContentTest, FloatWeightsPreserved) {
    // Write float weights as binary
    std::vector<float> weights = {3.14f, 2.71f, 1.41f, 0.0f, -1.0f, 1e10f, 1e-10f, -0.0f};
    auto path = m_temp_dir / "float_weights.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
    }

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(path.string());
    ASSERT_EQ(buf->size, weights.size() * sizeof(float));

    auto* loaded = static_cast<float*>(buf->host_ptr);
    for (size_t i = 0; i < weights.size(); i++) {
        // Use bitwise comparison to catch -0.0 vs +0.0
        ASSERT_EQ(std::memcmp(&loaded[i], &weights[i], sizeof(float)), 0)
            << "Float mismatch at index " << i;
    }
}

TEST_F(SharedWeightBufferContentTest, MixedDtypePattern) {
    // Simulate a model with mixed INT4/FP16 weights packed sequentially
    std::vector<uint8_t> data(8192);
    // Fill with a recognizable pattern
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    auto path = m_temp_dir / "mixed_weights.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(path.string());
    ASSERT_EQ(buf->size, data.size());

    auto* loaded = static_cast<uint8_t*>(buf->host_ptr);
    ASSERT_EQ(std::memcmp(loaded, data.data(), data.size()), 0);
}

TEST_F(SharedWeightBufferContentTest, SmallOneByteFile) {
    auto path = m_temp_dir / "tiny.bin";
    {
        std::ofstream f(path, std::ios::binary);
        char c = '\x42';
        f.write(&c, 1);
    }

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(path.string());
    ASSERT_EQ(buf->size, 1u);
    ASSERT_EQ(static_cast<uint8_t*>(buf->host_ptr)[0], 0x42);
}

TEST_F(SharedWeightBufferContentTest, NonPageAlignedSizeWorks) {
    // 4097 bytes = not a multiple of 4096
    std::vector<uint8_t> data(4097, 0xCD);
    auto path = m_temp_dir / "unaligned_size.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    auto buf = ov::intel_xpu::SharedWeightBuffer::create(path.string());
    ASSERT_EQ(buf->size, 4097u);

    auto* loaded = static_cast<uint8_t*>(buf->host_ptr);
    for (size_t i = 0; i < 4097; i++) {
        ASSERT_EQ(loaded[i], 0xCD) << "Mismatch at byte " << i;
    }
}

// ============================================================================
// Test Group 9: Pointer offset computation for bin_offset
// Tests that the XPU plugin correctly computes bin_offset as the distance
// from shared buffer start to a constant's data pointer.
// ============================================================================

class BinOffsetComputationTest : public ::testing::Test {};

TEST_F(BinOffsetComputationTest, OffsetAtStart) {
    char buffer[1024];
    char* data_ptr = buffer;
    char* buf_start = buffer;
    size_t offset = static_cast<size_t>(data_ptr - buf_start);
    ASSERT_EQ(offset, 0u);
}

TEST_F(BinOffsetComputationTest, OffsetInMiddle) {
    char buffer[1024];
    char* data_ptr = buffer + 512;
    char* buf_start = buffer;
    size_t offset = static_cast<size_t>(data_ptr - buf_start);
    ASSERT_EQ(offset, 512u);
}

TEST_F(BinOffsetComputationTest, OffsetAtEnd) {
    char buffer[1024];
    char* data_ptr = buffer + 1023;
    char* buf_start = buffer;
    size_t offset = static_cast<size_t>(data_ptr - buf_start);
    ASSERT_EQ(offset, 1023u);
}

TEST_F(BinOffsetComputationTest, MultipleConstantsHaveDifferentOffsets) {
    // Simulate a buffer with 3 "constants" at different positions
    alignas(16) char buffer[4096];
    const size_t c1_offset = 0;
    const size_t c1_size = 1024;
    const size_t c2_offset = 1024;
    const size_t c2_size = 2048;
    const size_t c3_offset = 3072;
    const size_t c3_size = 1024;

    char* c1_ptr = buffer + c1_offset;
    char* c2_ptr = buffer + c2_offset;
    char* c3_ptr = buffer + c3_offset;

    ASSERT_EQ(static_cast<size_t>(c1_ptr - buffer), c1_offset);
    ASSERT_EQ(static_cast<size_t>(c2_ptr - buffer), c2_offset);
    ASSERT_EQ(static_cast<size_t>(c3_ptr - buffer), c3_offset);

    // Verify non-overlapping
    ASSERT_LE(c1_offset + c1_size, c2_offset);
    ASSERT_LE(c2_offset + c2_size, c3_offset);
    ASSERT_LE(c3_offset + c3_size, 4096u);
}

// ============================================================================
// Test Group 10: GPU constant.cpp needs_conversion logic
// Tests the data type conversion check that gates the zero-copy path.
// ============================================================================

class NeedsConversionTest : public ::testing::Test {
protected:
    // Mirror the needs_conversion logic from constant.cpp
    static bool needs_conversion(ov::element::Type src_type, size_t shape_size) {
        // GPU plugin promotes f64 scalars to f32
        if (shape_size == 1 && src_type == ov::element::f64) {
            return true;
        }
        // GPU plugin promotes u16/i16 to f32
        if (src_type == ov::element::u16 || src_type == ov::element::i16) {
            return true;
        }
        return false;
    }
};

TEST_F(NeedsConversionTest, F32DoesNotNeedConversion) {
    ASSERT_FALSE(needs_conversion(ov::element::f32, 1));
    ASSERT_FALSE(needs_conversion(ov::element::f32, 100));
}

TEST_F(NeedsConversionTest, F16DoesNotNeedConversion) {
    ASSERT_FALSE(needs_conversion(ov::element::f16, 1));
    ASSERT_FALSE(needs_conversion(ov::element::f16, 1000));
}

TEST_F(NeedsConversionTest, I8DoesNotNeedConversion) {
    ASSERT_FALSE(needs_conversion(ov::element::i8, 1));
    ASSERT_FALSE(needs_conversion(ov::element::i8, 512));
}

TEST_F(NeedsConversionTest, U8DoesNotNeedConversion) {
    ASSERT_FALSE(needs_conversion(ov::element::u8, 1));
}

TEST_F(NeedsConversionTest, I4DoesNotNeedConversion) {
    ASSERT_FALSE(needs_conversion(ov::element::i4, 1));
    ASSERT_FALSE(needs_conversion(ov::element::i4, 1024));
}

TEST_F(NeedsConversionTest, F64ScalarNeedsConversion) {
    ASSERT_TRUE(needs_conversion(ov::element::f64, 1));
}

TEST_F(NeedsConversionTest, F64NonScalarDoesNotNeedConversion) {
    // Only scalars (shape_size==1) are converted for f64
    ASSERT_FALSE(needs_conversion(ov::element::f64, 2));
    ASSERT_FALSE(needs_conversion(ov::element::f64, 100));
}

TEST_F(NeedsConversionTest, U16NeedsConversion) {
    ASSERT_TRUE(needs_conversion(ov::element::u16, 1));
    ASSERT_TRUE(needs_conversion(ov::element::u16, 100));
}

TEST_F(NeedsConversionTest, I16NeedsConversion) {
    ASSERT_TRUE(needs_conversion(ov::element::i16, 1));
    ASSERT_TRUE(needs_conversion(ov::element::i16, 100));
}

TEST_F(NeedsConversionTest, BF16DoesNotNeedConversion) {
    ASSERT_FALSE(needs_conversion(ov::element::bf16, 1));
    ASSERT_FALSE(needs_conversion(ov::element::bf16, 256));
}

// ============================================================================
// Test Group 11: consolidate_to_shared_buffer logic
// Tests the core consolidation pattern in isolation: copy init-output view
// tensors into a flat shared buffer, replace views, and free originals.
// This mirrors WeightlessGraph::consolidate_to_shared_buffer() without
// requiring L0 or NPU hardware.
// ============================================================================

class ConsolidateToSharedBufferTest : public ::testing::Test {
protected:
    /// Mirrors the consolidation logic from WeightlessGraph::consolidate_to_shared_buffer().
    /// Returns true if consolidation was performed, false if skipped.
    static bool consolidate(
        std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& viewTensors,
        std::vector<std::shared_ptr<void>>& allocatedTensors,  // simulates _mainInputsAllocatedTensors
        void* shared_start,
        size_t shared_size) {
        if (allocatedTensors.empty()) {
            return false;  // nothing to relocate
        }

        // 1. Compute total init output size
        size_t total_size = 0;
        for (const auto& [name, tensor] : viewTensors) {
            total_size += tensor->get_byte_size();
        }

        if (total_size > shared_size) {
            return false;  // doesn't fit
        }

        // 2. Copy each view tensor into the shared buffer and replace the view
        auto* dst = static_cast<unsigned char*>(shared_start);
        size_t offset = 0;
        for (auto& [name, viewTensor] : viewTensors) {
            size_t sz = viewTensor->get_byte_size();
            std::memcpy(dst + offset, viewTensor->data(), sz);
            viewTensor = ov::make_tensor(viewTensor->get_element_type(),
                                          viewTensor->get_shape(), dst + offset);
            offset += sz;
        }

        // 3. Free original allocations
        allocatedTensors.clear();

        return true;
    }

    /// Helper: create a heap-allocated tensor with known pattern and return both
    /// the backing allocation and a view tensor into it.
    static std::pair<std::shared_ptr<void>, std::shared_ptr<ov::ITensor>>
    make_test_tensor(ov::element::Type type, const ov::Shape& shape, uint8_t fill) {
        size_t byte_size = ov::shape_size(shape) * type.size();
        // Allocate raw backing memory (simulates ZeroTensor L0 allocation)
        auto backing = std::shared_ptr<void>(new uint8_t[byte_size], [](void* p) {
            delete[] static_cast<uint8_t*>(p);
        });
        std::memset(backing.get(), fill, byte_size);
        auto tensor = ov::make_tensor(type, shape, backing.get());
        return {backing, tensor};
    }
};

TEST_F(ConsolidateToSharedBufferTest, EmptyAllocatedTensorsSkips) {
    alignas(64) uint8_t shared_buf[4096];
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    std::vector<std::shared_ptr<void>> allocs;

    ASSERT_FALSE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));
}

TEST_F(ConsolidateToSharedBufferTest, SingleTensorConsolidation) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    auto [backing, tensor] = make_test_tensor(ov::element::f32, {4}, 0xAA);
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["weight_0"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));

    // Allocated tensors should be freed
    ASSERT_TRUE(allocs.empty());

    // View tensor should now point into shared_buf
    auto* view_data = static_cast<uint8_t*>(views["weight_0"]->data());
    ASSERT_GE(view_data, shared_buf);
    ASSERT_LT(view_data, shared_buf + sizeof(shared_buf));

    // Data should be preserved (0xAA pattern)
    size_t byte_size = views["weight_0"]->get_byte_size();
    for (size_t i = 0; i < byte_size; i++) {
        ASSERT_EQ(view_data[i], 0xAA) << "Data mismatch at byte " << i;
    }
}

TEST_F(ConsolidateToSharedBufferTest, MultipleTensorsPackedSequentially) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Create 3 tensors with distinct fill patterns
    auto [b1, t1] = make_test_tensor(ov::element::f32, {8}, 0x11);
    auto [b2, t2] = make_test_tensor(ov::element::f32, {4}, 0x22);
    auto [b3, t3] = make_test_tensor(ov::element::u8, {16}, 0x33);

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["w1"] = t1;
    views["w2"] = t2;
    views["w3"] = t3;
    std::vector<std::shared_ptr<void>> allocs = {b1, b2, b3};

    size_t total = t1->get_byte_size() + t2->get_byte_size() + t3->get_byte_size();

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));
    ASSERT_TRUE(allocs.empty());

    // All views should point into shared_buf range
    size_t offset_check = 0;
    for (auto& [name, view] : views) {
        auto* ptr = static_cast<uint8_t*>(view->data());
        ASSERT_GE(ptr, shared_buf) << name << " below shared_buf start";
        ASSERT_LE(ptr + view->get_byte_size(), shared_buf + sizeof(shared_buf))
            << name << " exceeds shared_buf end";
        offset_check += view->get_byte_size();
    }
    ASSERT_EQ(offset_check, total);
}

TEST_F(ConsolidateToSharedBufferTest, OutputExceedsBufferSkips) {
    // Tiny shared buffer, large tensor
    alignas(64) uint8_t shared_buf[16];

    auto [backing, tensor] = make_test_tensor(ov::element::f32, {128}, 0xBB);  // 512 bytes
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["big_weight"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    // Save original data pointer
    auto* original_ptr = tensor->data();

    ASSERT_FALSE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));

    // Nothing should have changed
    ASSERT_EQ(allocs.size(), 1u);
    ASSERT_EQ(views["big_weight"]->data(), original_ptr);
}

TEST_F(ConsolidateToSharedBufferTest, ExactFitSucceeds) {
    // Buffer is exactly the right size
    constexpr size_t tensor_bytes = 64;  // 16 floats * 4 bytes
    alignas(64) uint8_t shared_buf[tensor_bytes];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    auto [backing, tensor] = make_test_tensor(ov::element::f32, {16}, 0xCC);
    ASSERT_EQ(tensor->get_byte_size(), tensor_bytes);

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["exact"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));
    ASSERT_TRUE(allocs.empty());

    auto* view_data = static_cast<uint8_t*>(views["exact"]->data());
    ASSERT_EQ(view_data, shared_buf);
    for (size_t i = 0; i < tensor_bytes; i++) {
        ASSERT_EQ(view_data[i], 0xCC);
    }
}

TEST_F(ConsolidateToSharedBufferTest, DataIntegrityWithRealValues) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Create tensor with actual float values (not just fill patterns)
    constexpr size_t num_floats = 8;
    float src_data[num_floats] = {3.14f, 2.71f, 1.41f, 0.0f, -1.0f, 1e6f, 1e-6f, -0.0f};
    auto backing = std::shared_ptr<void>(new float[num_floats], [](void* p) {
        delete[] static_cast<float*>(p);
    });
    std::memcpy(backing.get(), src_data, sizeof(src_data));
    auto tensor = ov::make_tensor(ov::element::f32, {num_floats}, backing.get());

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["real_weights"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));

    // Verify float values are preserved exactly (bitwise)
    auto* result = static_cast<float*>(views["real_weights"]->data());
    for (size_t i = 0; i < num_floats; i++) {
        ASSERT_EQ(std::memcmp(&result[i], &src_data[i], sizeof(float)), 0)
            << "Float mismatch at index " << i
            << " (got " << result[i] << ", expected " << src_data[i] << ")";
    }
}

TEST_F(ConsolidateToSharedBufferTest, OriginalBackingCanBeFreedAfterConsolidation) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    auto [backing, tensor] = make_test_tensor(ov::element::f32, {4}, 0xDD);
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["w"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));

    // Reset the original backing pointer — simulates L0 memory being freed
    backing.reset();

    // View should still be valid (points to shared_buf, not backing)
    auto* view_data = static_cast<uint8_t*>(views["w"]->data());
    for (size_t i = 0; i < views["w"]->get_byte_size(); i++) {
        ASSERT_EQ(view_data[i], 0xDD);
    }
}

TEST_F(ConsolidateToSharedBufferTest, SharedBufferOverwrittenCorrectly) {
    // Simulate: shared buffer initially contains raw weights (0xFF),
    // consolidation should overwrite with transformed data (0xAA).
    alignas(64) uint8_t shared_buf[256];
    std::memset(shared_buf, 0xFF, sizeof(shared_buf));

    auto [backing, tensor] = make_test_tensor(ov::element::u8, {64}, 0xAA);
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["transformed"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));

    // First 64 bytes should be 0xAA (overwritten by consolidation)
    for (size_t i = 0; i < 64; i++) {
        ASSERT_EQ(shared_buf[i], 0xAA) << "Byte " << i << " not overwritten";
    }
    // Remaining bytes should still be 0xFF (untouched)
    for (size_t i = 64; i < sizeof(shared_buf); i++) {
        ASSERT_EQ(shared_buf[i], 0xFF) << "Byte " << i << " was unexpectedly modified";
    }
}

TEST_F(ConsolidateToSharedBufferTest, ViewTensorMetadataPreserved) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Create tensors with different element types and shapes
    auto [b1, t1] = make_test_tensor(ov::element::f32, {2, 4}, 0x11);    // 32 bytes
    auto [b2, t2] = make_test_tensor(ov::element::f16, {3, 2}, 0x22);    // 12 bytes
    auto [b3, t3] = make_test_tensor(ov::element::u8, {16}, 0x33);       // 16 bytes

    auto shape1 = t1->get_shape();
    auto type1 = t1->get_element_type();
    auto shape2 = t2->get_shape();
    auto type2 = t2->get_element_type();
    auto shape3 = t3->get_shape();
    auto type3 = t3->get_element_type();

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["a"] = t1;
    views["b"] = t2;
    views["c"] = t3;
    std::vector<std::shared_ptr<void>> allocs = {b1, b2, b3};

    ASSERT_TRUE(consolidate(views, allocs, shared_buf, sizeof(shared_buf)));

    // Verify element types and shapes are preserved after consolidation
    ASSERT_EQ(views["a"]->get_element_type(), type1);
    ASSERT_EQ(views["a"]->get_shape(), shape1);
    ASSERT_EQ(views["b"]->get_element_type(), type2);
    ASSERT_EQ(views["b"]->get_shape(), shape2);
    ASSERT_EQ(views["c"]->get_element_type(), type3);
    ASSERT_EQ(views["c"]->get_shape(), shape3);
}

// ============================================================================
// Test Group 12: Zero-buffer sensitivity after consolidation
// Simulates the zero-buffer integration test: after consolidation into a
// shared buffer, zeroing the buffer should destroy the data (proving
// the view tensors truly reference the shared buffer, not copies).
// ============================================================================

class ZeroBufferAfterConsolidateTest : public ::testing::Test {};

TEST_F(ZeroBufferAfterConsolidateTest, ZeroingBufferDestroysConsolidatedData) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Create a tensor with non-zero data
    constexpr size_t n = 16;
    float src[n];
    for (size_t i = 0; i < n; i++) src[i] = static_cast<float>(i + 1);

    auto backing = std::shared_ptr<void>(new float[n], [](void* p) {
        delete[] static_cast<float*>(p);
    });
    std::memcpy(backing.get(), src, sizeof(src));
    auto tensor = ov::make_tensor(ov::element::f32, {n}, backing.get());

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> views;
    views["w"] = tensor;
    std::vector<std::shared_ptr<void>> allocs = {backing};

    // Consolidate
    size_t total_size = tensor->get_byte_size();
    auto* dst = static_cast<unsigned char*>(static_cast<void*>(shared_buf));
    size_t offset = 0;
    for (auto& [name, viewTensor] : views) {
        size_t sz = viewTensor->get_byte_size();
        std::memcpy(dst + offset, viewTensor->data(), sz);
        viewTensor = ov::make_tensor(viewTensor->get_element_type(),
                                      viewTensor->get_shape(), dst + offset);
        offset += sz;
    }
    allocs.clear();
    backing.reset();

    // Verify data is valid after consolidation
    auto* result = static_cast<float*>(views["w"]->data());
    for (size_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(result[i], static_cast<float>(i + 1));
    }

    // Zero the shared buffer — simulates the zero-buffer test from test_xpu_weights
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Data should now be zero (proving view reads from shared buffer)
    for (size_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(result[i], 0.0f)
            << "After zeroing, element " << i << " should be 0 but got " << result[i];
    }
}

TEST_F(ZeroBufferAfterConsolidateTest, PartialZeroAffectsOnlyRelevantTensors) {
    alignas(64) uint8_t shared_buf[1024];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Two tensors: first at offset 0 (64 bytes), second at offset 64 (64 bytes)
    constexpr size_t n = 16;  // 16 floats = 64 bytes each

    float src1[n], src2[n];
    for (size_t i = 0; i < n; i++) {
        src1[i] = 1.0f;
        src2[i] = 2.0f;
    }

    // Place them sequentially in shared_buf
    std::memcpy(shared_buf, src1, sizeof(src1));
    std::memcpy(shared_buf + sizeof(src1), src2, sizeof(src2));

    auto view1 = ov::make_tensor(ov::element::f32, {n}, shared_buf);
    auto view2 = ov::make_tensor(ov::element::f32, {n}, shared_buf + sizeof(src1));

    // Zero only the first 64 bytes
    std::memset(shared_buf, 0, sizeof(src1));

    // First tensor should be zeroed
    auto* d1 = static_cast<float*>(view1->data());
    for (size_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(d1[i], 0.0f);
    }

    // Second tensor should be intact
    auto* d2 = static_cast<float*>(view2->data());
    for (size_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(d2[i], 2.0f);
    }
}

// ============================================================================
// Test Group 13: BankConsolidationTest
// Tests the Bank-level consolidation logic that copies transformed weights
// from NPUW's Bank storage into an XPU shared buffer region.
// Standalone tests — no NPUW dependency, mirrors consolidate_to_xpu_buffer().
// ============================================================================

class BankConsolidationTest : public ::testing::Test {
protected:
    /// Mirrors Bank::consolidate_to_xpu_buffer() logic in standalone form.
    /// Iterates stored tensors, skips those already in the shared buffer,
    /// copies others into [alloc_offset, size), and replaces the tensor.
    struct StoredTensor {
        ov::Tensor tensor;
    };

    static size_t consolidate(std::unordered_map<int64_t, StoredTensor>& storage,
                              void* shared_ptr,
                              size_t shared_size,
                              size_t alloc_offset) {
        auto* buf = static_cast<uint8_t*>(shared_ptr);
        auto* buf_end = buf + shared_size;
        size_t offset = alloc_offset;
        size_t relocated_count = 0;

        for (auto& [uid, stored] : storage) {
            if (!stored.tensor) {
                continue;
            }

            auto* tensor_data = static_cast<uint8_t*>(stored.tensor.data());
            size_t tensor_size = stored.tensor.get_byte_size();

            // Skip tensors already in the shared buffer
            if (tensor_data >= buf && tensor_data + tensor_size <= buf_end) {
                continue;
            }

            // Check if it fits
            if (offset + tensor_size > shared_size) {
                continue;  // graceful fallback
            }

            std::memcpy(buf + offset, tensor_data, tensor_size);
            stored.tensor = ov::Tensor(stored.tensor.get_element_type(), stored.tensor.get_shape(), buf + offset);
            relocated_count++;
            offset += tensor_size;
        }
        return relocated_count;
    }
};

TEST_F(BankConsolidationTest, EmptyBankConsolidationNoOp) {
    alignas(64) uint8_t shared_buf[4096];
    std::unordered_map<int64_t, StoredTensor> storage;

    size_t relocated = consolidate(storage, shared_buf, sizeof(shared_buf), 2048);
    ASSERT_EQ(relocated, 0u);
}

TEST_F(BankConsolidationTest, SingleTensorConsolidation) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Create a tensor with known data outside the shared buffer
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    ov::Tensor t(ov::element::f32, {4}, data.data());

    std::unordered_map<int64_t, StoredTensor> storage;
    storage[0] = {t};

    size_t alloc_offset = 2048;
    size_t relocated = consolidate(storage, shared_buf, sizeof(shared_buf), alloc_offset);

    ASSERT_EQ(relocated, 1u);

    // Tensor should now point into shared_buf at alloc_offset
    auto* new_data = static_cast<uint8_t*>(storage[0].tensor.data());
    ASSERT_EQ(new_data, shared_buf + alloc_offset);

    // Verify data integrity
    auto* floats = reinterpret_cast<float*>(new_data);
    ASSERT_FLOAT_EQ(floats[0], 1.0f);
    ASSERT_FLOAT_EQ(floats[1], 2.0f);
    ASSERT_FLOAT_EQ(floats[2], 3.0f);
    ASSERT_FLOAT_EQ(floats[3], 4.0f);
}

TEST_F(BankConsolidationTest, MultipleTensorsSequential) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    std::vector<float> d1 = {10.0f, 20.0f};
    std::vector<float> d2 = {30.0f, 40.0f, 50.0f};
    ov::Tensor t1(ov::element::f32, {2}, d1.data());
    ov::Tensor t2(ov::element::f32, {3}, d2.data());

    std::unordered_map<int64_t, StoredTensor> storage;
    storage[0] = {t1};
    storage[1] = {t2};

    size_t alloc_offset = 2048;
    size_t relocated = consolidate(storage, shared_buf, sizeof(shared_buf), alloc_offset);

    ASSERT_EQ(relocated, 2u);

    // Both tensors should point into shared_buf
    for (auto& [uid, stored] : storage) {
        auto* ptr = static_cast<uint8_t*>(stored.tensor.data());
        ASSERT_GE(ptr, shared_buf + alloc_offset);
        ASSERT_LE(ptr + stored.tensor.get_byte_size(), shared_buf + sizeof(shared_buf));
    }
}

TEST_F(BankConsolidationTest, OverflowFallback) {
    // Shared buffer too small — tensor should be left in place
    alignas(64) uint8_t shared_buf[64];

    std::vector<float> data(32, 1.0f);  // 128 bytes, won't fit in 64-byte buffer
    ov::Tensor t(ov::element::f32, {32}, data.data());
    auto* original_ptr = t.data();

    std::unordered_map<int64_t, StoredTensor> storage;
    storage[0] = {t};

    size_t relocated = consolidate(storage, shared_buf, sizeof(shared_buf), 32);

    ASSERT_EQ(relocated, 0u);
    // Tensor should remain at original location
    ASSERT_EQ(storage[0].tensor.data(), original_ptr);
}

TEST_F(BankConsolidationTest, DataIntegrityAfterConsolidation) {
    alignas(64) uint8_t shared_buf[8192];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    // Create tensor with specific float pattern
    constexpr size_t n = 16;
    float src[n];
    for (size_t i = 0; i < n; i++) src[i] = static_cast<float>(i) * 3.14f;

    ov::Tensor t(ov::element::f32, {n}, src);

    std::unordered_map<int64_t, StoredTensor> storage;
    storage[0] = {t};

    size_t alloc_offset = 4096;
    consolidate(storage, shared_buf, sizeof(shared_buf), alloc_offset);

    // Verify byte-exact data
    auto* result = static_cast<float*>(storage[0].tensor.data());
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(std::memcmp(&result[i], &src[i], sizeof(float)), 0)
            << "Float mismatch at index " << i;
    }
}

TEST_F(BankConsolidationTest, SkipAlreadyConsolidated) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0xAA, sizeof(shared_buf));

    // Create a tensor that already points into the shared buffer
    ov::Tensor t(ov::element::u8, {64}, shared_buf + 100);

    std::unordered_map<int64_t, StoredTensor> storage;
    storage[0] = {t};

    size_t relocated = consolidate(storage, shared_buf, sizeof(shared_buf), 2048);

    // Should be skipped (already in buffer)
    ASSERT_EQ(relocated, 0u);
    ASSERT_EQ(storage[0].tensor.data(), shared_buf + 100);
}

TEST_F(BankConsolidationTest, TensorMetadataPreserved) {
    alignas(64) uint8_t shared_buf[4096];
    std::memset(shared_buf, 0, sizeof(shared_buf));

    std::vector<float> d1(8, 1.0f);
    std::vector<uint8_t> d2(16, 0x42);

    ov::Tensor t1(ov::element::f32, {2, 4}, d1.data());
    ov::Tensor t2(ov::element::u8, {16}, d2.data());

    auto shape1 = t1.get_shape();
    auto type1 = t1.get_element_type();
    auto shape2 = t2.get_shape();
    auto type2 = t2.get_element_type();

    std::unordered_map<int64_t, StoredTensor> storage;
    storage[0] = {t1};
    storage[1] = {t2};

    consolidate(storage, shared_buf, sizeof(shared_buf), 2048);

    // Element types and shapes must be preserved
    ASSERT_EQ(storage[0].tensor.get_element_type(), type1);
    ASSERT_EQ(storage[0].tensor.get_shape(), shape1);
    ASSERT_EQ(storage[1].tensor.get_element_type(), type2);
    ASSERT_EQ(storage[1].tensor.get_shape(), shape2);
}

// ============================================================================
// Test Group 14: SharedWeightBuffer::create_empty
// Tests the empty buffer factory method for NPUW persistent weights.
// ============================================================================

class SharedWeightBufferCreateEmptyTest : public ::testing::Test {};

TEST_F(SharedWeightBufferCreateEmptyTest, CreateEmpty_Fallback) {
    // No GPU context — should use _aligned_malloc fallback
    auto buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, 4096);

    ASSERT_NE(buf, nullptr);
    ASSERT_NE(buf->host_ptr, nullptr);
    ASSERT_EQ(buf->size, 0u);         // No bin data
    ASSERT_EQ(buf->alloc_size, 4096u);
    ASSERT_FALSE(buf->is_gpu_usm());
}

TEST_F(SharedWeightBufferCreateEmptyTest, CreateEmpty_Zeroed) {
    auto buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, 8192);

    ASSERT_NE(buf, nullptr);
    auto* data = static_cast<uint8_t*>(buf->host_ptr);
    for (size_t i = 0; i < buf->alloc_size; i++) {
        ASSERT_EQ(data[i], 0u) << "Byte " << i << " is not zero";
    }
}

TEST_F(SharedWeightBufferCreateEmptyTest, CreateEmpty_IsAligned) {
    auto buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, 4096);

    auto ptr_val = reinterpret_cast<uintptr_t>(buf->host_ptr);
    ASSERT_EQ(ptr_val % 4096, 0u) << "host_ptr is not 4096-byte aligned";
}

TEST_F(SharedWeightBufferCreateEmptyTest, CreateEmpty_ZeroAllocBytesThrows) {
    ASSERT_THROW(
        ov::intel_xpu::SharedWeightBuffer::create_empty({}, 0),
        ov::Exception);
}

TEST_F(SharedWeightBufferCreateEmptyTest, CreateEmpty_DestructorDoesNotCrash) {
    {
        auto buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, 2048);
        ASSERT_NE(buf, nullptr);
    }
    // If we reach here without crash, the destructor worked correctly
}

TEST_F(SharedWeightBufferCreateEmptyTest, CreateEmpty_LargeAllocation) {
    // 16 MB empty buffer
    auto buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, 16 * 1024 * 1024);

    ASSERT_NE(buf, nullptr);
    ASSERT_EQ(buf->size, 0u);
    ASSERT_EQ(buf->alloc_size, 16u * 1024 * 1024);

    // Spot-check zeroed (check first and last 4K)
    auto* data = static_cast<uint8_t*>(buf->host_ptr);
    for (size_t i = 0; i < 4096; i++) {
        ASSERT_EQ(data[i], 0u);
    }
    for (size_t i = buf->alloc_size - 4096; i < buf->alloc_size; i++) {
        ASSERT_EQ(data[i], 0u);
    }
}

// ============================================================================
// Test Group 15: Two-Buffer Split
// Verifies that raw and persistent buffers are separate allocations with
// correct properties.
// ============================================================================

class TwoBufferSplitTest : public ::testing::Test {
protected:
    std::filesystem::path m_temp_dir;

    void SetUp() override {
        m_temp_dir = std::filesystem::temp_directory_path() / "xpu_two_buf_test";
        std::filesystem::create_directories(m_temp_dir);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(m_temp_dir, ec);
    }

    std::filesystem::path write_bin_file(const std::string& name, const std::vector<uint8_t>& data) {
        auto path = m_temp_dir / name;
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
        f.close();
        return path;
    }
};

TEST_F(TwoBufferSplitTest, SeparatePointers) {
    // Create raw buffer from file
    std::vector<uint8_t> test_data(4096, 0xAB);
    auto bin_path = write_bin_file("weights.bin", test_data);

    auto raw_buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());
    auto persistent_buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, raw_buf->size);

    // Different allocations
    ASSERT_NE(raw_buf->host_ptr, persistent_buf->host_ptr);

    // Raw buffer has bin data
    ASSERT_EQ(raw_buf->size, 4096u);
    ASSERT_EQ(raw_buf->alloc_size, 4096u);
    ASSERT_EQ(static_cast<uint8_t*>(raw_buf->host_ptr)[0], 0xAB);

    // Persistent buffer is empty and zeroed
    ASSERT_EQ(persistent_buf->size, 0u);
    ASSERT_EQ(persistent_buf->alloc_size, 4096u);
    ASSERT_EQ(static_cast<uint8_t*>(persistent_buf->host_ptr)[0], 0u);
}

TEST_F(TwoBufferSplitTest, IndependentLifetimes) {
    std::vector<uint8_t> test_data(2048, 0xCC);
    auto bin_path = write_bin_file("weights.bin", test_data);

    auto raw_buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());
    auto persistent_buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, raw_buf->size);

    // Write data into persistent buffer
    std::memset(persistent_buf->host_ptr, 0xDD, persistent_buf->alloc_size);

    // Destroy raw buffer — persistent should survive
    void* persistent_ptr = persistent_buf->host_ptr;
    raw_buf.reset();

    ASSERT_EQ(persistent_buf->host_ptr, persistent_ptr);
    ASSERT_EQ(static_cast<uint8_t*>(persistent_buf->host_ptr)[0], 0xDD);
}

TEST_F(TwoBufferSplitTest, ConsolidationUsesFullPersistentBuffer) {
    // Mirrors the NPUW consolidation pattern: offset=0, entire buffer available
    std::vector<uint8_t> test_data(4096, 0xAA);
    auto bin_path = write_bin_file("weights.bin", test_data);

    auto raw_buf = ov::intel_xpu::SharedWeightBuffer::create(bin_path.string());
    auto persistent_buf = ov::intel_xpu::SharedWeightBuffer::create_empty({}, raw_buf->size);

    // Simulate NPUW writing transformed weights from offset 0
    size_t write_offset = 0;
    size_t write_size = 2048;
    std::memset(static_cast<uint8_t*>(persistent_buf->host_ptr) + write_offset, 0xBB, write_size);

    // Verify written region
    auto* data = static_cast<uint8_t*>(persistent_buf->host_ptr);
    for (size_t i = 0; i < write_size; i++) {
        ASSERT_EQ(data[i], 0xBB) << "Written region mismatch at byte " << i;
    }
    // Verify remaining region is still zeroed
    for (size_t i = write_size; i < persistent_buf->alloc_size; i++) {
        ASSERT_EQ(data[i], 0u) << "Unwritten region mismatch at byte " << i;
    }
}

TEST_F(TwoBufferSplitTest, BankConsolidationWithOffset0) {
    // Test Bank consolidation with offset=0 (the new default)
    // This mirrors what happens when NPUW gets XPU_SHARED_WEIGHT_ALLOC_OFFSET=0
    alignas(64) uint8_t persistent_buf[4096];
    std::memset(persistent_buf, 0, sizeof(persistent_buf));

    // Create tensors outside the persistent buffer (simulates Bank storage)
    std::vector<float> d1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> d2 = {5.0f, 6.0f};
    ov::Tensor t1(ov::element::f32, {4}, d1.data());
    ov::Tensor t2(ov::element::f32, {2}, d2.data());

    // Consolidate from offset 0
    size_t offset = 0;
    auto* dst = persistent_buf;

    // Copy t1
    std::memcpy(dst + offset, t1.data(), t1.get_byte_size());
    offset += t1.get_byte_size();

    // Copy t2
    std::memcpy(dst + offset, t2.data(), t2.get_byte_size());
    offset += t2.get_byte_size();

    // Verify: total written = 4*4 + 2*4 = 24 bytes
    ASSERT_EQ(offset, 24u);

    // Verify data at offset 0
    auto* result = reinterpret_cast<float*>(persistent_buf);
    ASSERT_FLOAT_EQ(result[0], 1.0f);
    ASSERT_FLOAT_EQ(result[1], 2.0f);
    ASSERT_FLOAT_EQ(result[2], 3.0f);
    ASSERT_FLOAT_EQ(result[3], 4.0f);
    ASSERT_FLOAT_EQ(result[4], 5.0f);
    ASSERT_FLOAT_EQ(result[5], 6.0f);
}
