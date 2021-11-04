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
 * \brief Constant 层描述创建器
 */
template <>
class TLayerDescCreator<TrtConstantDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();
    // 在这里补充其他的常量算子
    return kind == c10::aten::arange || kind == c10::aten::zeros || kind == c10::aten::ones ||
           kind == c10::prim::GetAttr;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    input_values.push_back(nullptr);

    const auto kind = node->kind();
    at::Tensor tensor;
    if (kind == c10::aten::arange) {
      tensor = CreateConstantFromArrange(module, node);
    } else if (kind == c10::aten::zeros || kind == c10::aten::ones) {
      tensor = CreateConstantFromZerosOrOnes(module, node);
    } else if (kind == c10::prim::GetAttr) {
      tensor = CreateConstantFromGetAttr(module, node);
    }

    auto layer_desc = std::make_shared<TrtConstantDesc>();
    layer_desc->weights = ToFwdWeights(tensor);
    layer_desc->dimensions = DimsOf(tensor);
    return layer_desc;
  }

 private:
  at::Tensor CreateConstantFromGetAttr(const TorchModule& module, const JitNode* node) {
    LOG(INFO) << "TrtConstantDesc::CreateGetAttr";
    const auto output = module.Get(node->output());

    if (output.isTensor()) return output.toTensor();
    if (output.isScalar()) return output.toScalar() * torch::ones(1);

    LOG(ERROR) << "Unsupported IValue type. Only support : Tensor and Scalar.";
    return {};
  }

  at::Tensor CreateConstantFromZerosOrOnes(const TorchModule& module, const JitNode* node) {
    LOG(INFO) << "TrtConstantDesc::CreateZerosOrOnes";
    const auto inputs = node->inputs();
    const auto size = module.Get(inputs[0]).toIntVector();

    if (node->kind() == c10::aten::zeros) return torch::zeros(size);

    return torch::ones(size);
  }

  at::Tensor CreateConstantFromArrange(const TorchModule& module, const JitNode* node) {
    LOG(INFO) << "TrtConstantDesc::CreateArrange";
    const auto inputs = node->inputs();

    if (inputs.size() == 5) {
      const c10::Scalar end = module.Get(inputs[0]).toScalar();
      return torch::arange(end);
    }

    const c10::Scalar start = module.Get(inputs[0]).toScalar();
    const c10::Scalar end = module.Get(inputs[1]).toScalar();
    const c10::Scalar step = module.Get(inputs[2]).toScalar();
    return torch::arange(start, end, step);
  }
};

FWD_TORCH_NAMESPACE_END
