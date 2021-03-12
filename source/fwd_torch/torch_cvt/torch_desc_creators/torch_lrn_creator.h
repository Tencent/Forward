// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.
//
// ╔════════════════════════════════════════════════════════════════════════════════════════╗
// ║──█████████╗───███████╗───████████╗───██╗──────██╗───███████╗───████████╗───████████╗───║
// ║──██╔══════╝──██╔════██╗──██╔════██╗──██║──────██║──██╔════██╗──██╔════██╗──██╔════██╗──║
// ║──████████╗───██║────██║──████████╔╝──██║──█╗──██║──█████████║──████████╔╝──██║────██║──║
// ║──██╔═════╝───██║────██║──██╔════██╗──██║█████╗██║──██╔════██║──██╔════██╗──██║────██║──║
// ║──██║─────────╚███████╔╝──██║────██║──╚████╔████╔╝──██║────██║──██║────██║──████████╔╝──║
// ║──╚═╝──────────╚══════╝───╚═╝────╚═╝───╚═══╝╚═══╝───╚═╝────╚═╝──╚═╝────╚═╝──╚═══════╝───║
// ╚════════════════════════════════════════════════════════════════════════════════════════╝
//
// Authors: Aster JIAN (asterjian@qq.com)
//          Yzx (yzxyzxyzx777@outlook.com)
//          Ao LI (346950981@qq.com)
//          Paul LU (lujq96@gmail.com)

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief LRN 层描述创建器
 */
template <>
class TLayerDescCreator<TrtLRNDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    // LOG(INFO) << "TrtLRNDesc::Check";

    // 使用 pass 自定义 fwd::lrn op之后，可以通过这个 op 来判断
    if (node->kind() == c10::Symbol::fromQualString("fwd::lrn")) {
      return true;
    }

    // 当前(v1.3.0)Jit模型会把lrn操作拆分成十余节点，需要链式判断节点类型是否吻合
    // 检查 c10::aten::div 节点
    if (!(node->kind() == c10::aten::div)) {
      return false;
    }

    // 检查 c10::aten::pow 节点
    auto child = node->inputs()[1]->node();
    if (!(child->kind() == c10::aten::pow)) {
      return false;
    }

    // 检查其他节点类型
    {
      const std::vector<c10::Symbol> kinds{
          c10::aten::add,     c10::aten::mul,        c10::aten::view,
          c10::aten::squeeze, c10::aten::avg_pool3d, c10::aten::constant_pad_nd,
          c10::aten::view,    c10::aten::unsqueeze,  c10::aten::mul,
      };

      for (const auto kind : kinds) {
        child = child->inputs()[0]->node();
        if (!(child->kind() == kind)) {
          return false;
        }
      }
    }

    // 检查输入是否相等
    if (node->inputs()[0] != child->inputs()[0]) {
      return false;
    }

    return true;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtLRNDesc::Create";
    auto layer_desc = std::make_shared<TrtLRNDesc>();

    input_values.push_back(node->inputs()[0]);

    if (node->kind() == c10::Symbol::fromQualString("fwd::lrn")) {
      layer_desc->windowSize = static_cast<int>(module.Get(node->inputs()[1]).toInt());
      layer_desc->alpha =
          static_cast<float>(module.Get(node->inputs()[2]).toTensor().item().toDouble());
      layer_desc->beta = static_cast<float>(module.Get(node->inputs()[3]).toDouble());
      layer_desc->k =
          static_cast<float>(module.Get(node->inputs()[4]).toTensor().item().toDouble());
    } else {
      // 从链上的节点中获取有用的参数
      const auto pow_node = node->inputs()[1]->node();
      const auto add_node = pow_node->inputs()[0]->node();
      const auto mul_node = add_node->inputs()[0]->node();
      const auto view_node = mul_node->inputs()[0]->node();
      const auto squeeze_node = view_node->inputs()[0]->node();
      const auto avg_pool3d_node = squeeze_node->inputs()[0]->node();

      const auto output_size = module.Get(avg_pool3d_node->inputs()[1]).toIntList();
      layer_desc->windowSize = static_cast<int>(output_size[0]);

      const auto alpha = module.Get(mul_node->inputs()[1]).toTensor().item().toDouble();
      layer_desc->alpha = static_cast<float>(alpha);

      const auto beta = module.Get(pow_node->inputs()[1]).toDouble();
      layer_desc->beta = static_cast<float>(beta);

      const auto k = module.Get(add_node->inputs()[1]).toTensor().item().toInt();
      layer_desc->k = k;
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
