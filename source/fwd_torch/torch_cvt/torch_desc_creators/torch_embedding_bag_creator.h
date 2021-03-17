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
 * \brief ReflectionPadding 层描述创建器
 */
template <>
class TLayerDescCreator<TrtEmbeddingBagDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::embedding_bag;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtEmbeddingBagDescCreator::Create";

    auto layer_desc = std::make_shared<TrtEmbeddingBagDesc>();

    const auto inputs = node->inputs();
    // Input 0, kind = Tensor weight
    // Input 1, kind = Tensor indices
    // Input 2, kind = Tensor offset
    // Input 3, kind = bool scale_grad_by_freq
    // Input 4, kind = int64_t mode;
    // Input 5 - 7(8) not supported yet
    input_values.push_back(inputs[1]);
    input_values.push_back(inputs[2]);

    const auto mode = module.Get(inputs[4]).toInt();
    layer_desc->op = mode;
    const auto weight = module.Get(inputs[0]).toTensor();
    layer_desc->count = weight.sizes()[0];
    layer_desc->dim = weight.sizes()[1];
    layer_desc->offset = 0;
    layer_desc->data = ToFwdWeights(weight);

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
