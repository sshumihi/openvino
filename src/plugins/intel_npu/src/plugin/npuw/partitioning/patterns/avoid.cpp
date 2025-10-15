// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avoid.hpp"
#include <openvino/op/op.hpp>

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace avoid {

namespace opp = ov::pass::pattern;

//------------------------------------------------------------------------------
// Pattern: RMSNorm, from LLaMa-v2-7b model
//
//            Power     Const
//               :        :
//               V        V
//               ReduceMean
//                    :
//                    V
//                   Add
//                    :
//                    V
//                   Sqrt
//                    :
//                    V
//
RMSNorm::RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto power = opp::wrap_type<ov::op::v1::Power>({opp::any_input(), opp::any_input()});
    auto reduce = opp::wrap_type<ov::op::v1::ReduceMean>({power, opp::wrap_type<ov::op::v0::Constant>()});
    auto add = opp::wrap_type<ov::op::v1::Add>({reduce, opp::any_input()});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({add});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_power = node_to_output.at(power).get_node_shared_ptr();
        auto matched_reduce = node_to_output.at(reduce).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();

        node_to_gptr->at(matched_power)->avoid(avoid_device);
        node_to_gptr->at(matched_reduce)->avoid(avoid_device);
        node_to_gptr->at(matched_add)->avoid(avoid_device);
        node_to_gptr->at(matched_sqrt)->avoid(avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sqrt, "TagRMSNormAvoid"), std::move(callback));
}

// From DeepSeek
SinCos::SinCos(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({opp::wrap_type<ov::op::v0::Constant>(), concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({opp::any_input(), opp::wrap_type<ov::op::v0::Constant>()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({concat_2});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_shape_of = node_to_output.at(shape_of).get_node_shared_ptr();
        auto matched_gather = node_to_output.at(gather).get_node_shared_ptr();
        auto matched_concat_1 = node_to_output.at(concat_1).get_node_shared_ptr();
        auto matched_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
        auto matched_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
        auto matched_convert = node_to_output.at(convert).get_node_shared_ptr();
        auto matched_matmul = node_to_output.at(matmul).get_node_shared_ptr();
        auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
        auto matched_concat_2 = node_to_output.at(concat_2).get_node_shared_ptr();
        auto matched_sin_cos = node_to_output.at(sin_cos).get_node_shared_ptr();

        node_to_gptr->at(matched_shape_of)->avoid(avoid_device);
        node_to_gptr->at(matched_gather)->avoid(avoid_device);
        node_to_gptr->at(matched_concat_1)->avoid(avoid_device);
        node_to_gptr->at(matched_broadcast)->avoid(avoid_device);
        node_to_gptr->at(matched_unsqueeze)->avoid(avoid_device);
        node_to_gptr->at(matched_convert)->avoid(avoid_device);
        node_to_gptr->at(matched_matmul)->avoid(avoid_device);
        node_to_gptr->at(matched_transpose)->avoid(avoid_device);
        node_to_gptr->at(matched_concat_2)->avoid(avoid_device);
        node_to_gptr->at(matched_sin_cos)->avoid(avoid_device);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sin_cos, "TagSinCos"), std::move(callback));
}

DynamicNodes::DynamicNodes(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                           const std::string& avoid_device) {
    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto dynamic_op = opp::wrap_type<ov::op::Op>([node_to_gptr](const std::shared_ptr<ov::Node>& node) {
        if (!node_to_gptr || node_to_gptr->find(node) == node_to_gptr->end()) {
            return false;
        }

        const auto inputs_count = node->get_input_size();
        return inputs_count > 0 && inputs_count <= 7;
    });

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_dynamic_op = node_to_output.at(dynamic_op).get_node_shared_ptr();

        const auto map_it = node_to_gptr->find(matched_dynamic_op);
        if (map_it == node_to_gptr->end()) {
            return false;
        }

        const auto outputs_count = matched_dynamic_op->get_output_size();
        for (size_t output_idx = 0; output_idx < outputs_count; ++output_idx) {
            if (matched_dynamic_op->get_output_partial_shape(output_idx).is_dynamic()) {
                map_it->second->avoid(avoid_device);
                break;
            }
        }

        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(dynamic_op, "TagDynamicNodesAvoid"), std::move(callback));
}



}  // namespace avoid
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
