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
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Norm 范数层描述创建器
 */
template <>
class TLayerDescCreator<TrtNormDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();

    // TODO(Paul Lu): 评估要不要做这个判断
    if (kind == c10::aten::div) {
      const JitNode* child = node->inputs()[1]->node();
      return !child->inputs().empty() && child->inputs()[0] == node->inputs()[0] &&
             child->kind() == c10::aten::norm;
    }

    // TODO(Paul Lu): 处理Frobenius范数
    return kind == c10::aten::norm;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtNormDesc::Create";

    auto layer_desc = std::make_shared<TrtNormDesc>();

    c10::ArrayRef<const JitValue*> inputs;
    if (node->kind() == c10::aten::norm) {
      inputs = node->inputs();
      layer_desc->div = false;
    } else {
      inputs = node->inputs()[1]->node()->inputs();
      layer_desc->div = true;
    }

    // Input 0, kind = input
    // Input 1 kind = (optional)c10::prim::Constant p
    // Input 2 kind = c10::prim::ListConstruct dim
    // Input 3 kind = c10::prim::Constant keepDim
    // (Input 4 kind = ScalarType dtype)
    input_values.push_back(inputs[0]);

    float p = 2.0f;
    if (!module.Get(inputs[1]).isNone()) {
      p = module.Get(inputs[1]).toScalar().toDouble();
    }
    layer_desc->p = p;
    layer_desc->inv_p = 1 / p;

    const auto dims = module.Get(inputs[2]).toIntList();
    int64_t axes = 0;
    for (auto it = dims.begin(); it != dims.end(); ++it) {
      axes |= 1 << *it;
    }
    layer_desc->axes = axes;

    layer_desc->keepDim = module.Get(inputs[3]).toBool();

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
