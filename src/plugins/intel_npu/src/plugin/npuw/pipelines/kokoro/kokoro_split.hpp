// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kokoro_compiled_model.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"

namespace ov {
class Node;
}

namespace ov {
namespace npuw {

class KokoroSplit {
public:
    static KokoroSplitResult split_model(const std::shared_ptr<ov::Model>& model, const KokoroConfig& config);

    static void serialize_model_a(const std::shared_ptr<ov::Model>& model_a, const std::string& output_dir);
    static void serialize_model_b(const std::shared_ptr<ov::Model>& model_b, const std::string& output_dir);
    
private:
    // Create model A - up to pred_dur output
    static std::shared_ptr<ov::Model> create_model_a(const std::shared_ptr<ov::Model>& model,
                                                     const KokoroConfig& config);

    // Create model B - from pred_dur output to the end
    static std::shared_ptr<ov::Model> create_model_b(const std::shared_ptr<ov::Model>& model,
                                                     const KokoroConfig& config);

    static std::shared_ptr<ov::Node> find_pred_dur_node(const std::shared_ptr<ov::Model>& model);
    static std::shared_ptr<ov::Node> find_en_matmul_node(const std::shared_ptr<ov::Model>& model);
    static std::shared_ptr<ov::Node> find_asr_matmul_node(const std::shared_ptr<ov::Model>& model);

    
};

}  // namespace npuw
}  // namespace ov
