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
#include <unordered_map>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"

FWD_TORCH_NAMESPACE_BEGIN

// ElementWise Description Creator
template <>
class TLayerDescCreator<TrtElementWiseDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();
    return NK2EWP_MAPPING.find(kind) != NK2EWP_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtElementWiseDesc::Create";

    const auto inputs = node->inputs();
    const auto kind = node->kind();
    T_CHECK_GE(inputs.size(), 2);
    // Input 0: Tensor input0
    // Input 1: Tensor input1
    // Input 2: Scalar alpha, add, add_, sub, rsub

    // op = rsub 读入顺序为 inputs[1], inputs[0]; 否则为inputs[0], inputs[1]
    const bool isRSub = (kind == c10::aten::rsub);

    auto layer_desc = std::make_shared<TrtElementWiseDesc>();
    // Inputs can be constant
    for (int i = 0; i < 2; i++) {
      auto iv = inputs[i ^ isRSub];
      if (iv->node()->kind().is_prim() || iv->node()->kind() == c10::aten::size) {
        if (!CreateConstantInput(module, iv, layer_desc->inputs[i])) return {};
      } else {
        input_values.push_back(iv);
      }
    }

    if (input_values.empty()) input_values.push_back(nullptr);

    // Tensor op Scalar 时，可能会出现alpha != 1, 系数与Scalar相乘
    const float alpha = inputs.size() > 2 ? module.Get(inputs[2]).toInt() : 1;
    if (alpha != 1) {
      auto& static_input = layer_desc->inputs[1 ^ isRSub].data;
      const float value = *(reinterpret_cast<const float*>(static_input.Data()));
      static_input = ToFwdWeights(value * alpha);
    }

    layer_desc->operation = NK2EWP_MAPPING.find(kind)->second;

    return layer_desc;
  }

 private:
  bool CreateConstantInput(const TorchModule& module, const torch::jit::Value* iv,
                           ConstantInput& c_input) const {
    const auto& input = module.Get(iv);
    at::Tensor tensor = torch::ones(1);
    if (input.isTensor()) {
      tensor = input.toTensor();
    } else if (input.isScalar()) {
      tensor = input.toScalar() * tensor;
    } else {
      LOG(ERROR) << "Create ConstantInput : Input Type " << input.type()->str()
                 << " Not Supported yet.";
      LOG(ERROR) << "Create Desc Failed.";
      return false;
    }
    c_input.inUse = true;
    c_input.data = ToFwdWeights(tensor);
    c_input.dim = DimsOf(tensor);
    return true;
  }

  const std::unordered_map<c10::Symbol, nvinfer1::ElementWiseOperation> NK2EWP_MAPPING = {
      {c10::aten::add, nvinfer1::ElementWiseOperation::kSUM},             // 0
      {c10::aten::add_, nvinfer1::ElementWiseOperation::kSUM},            // 0
      {c10::aten::mul, nvinfer1::ElementWiseOperation::kPROD},            // 1
      {c10::aten::mul_, nvinfer1::ElementWiseOperation::kPROD},           // 1
      {c10::aten::sub, nvinfer1::ElementWiseOperation::kSUB},             // 4
      {c10::aten::sub_, nvinfer1::ElementWiseOperation::kSUB},            // 4
      {c10::aten::rsub, nvinfer1::ElementWiseOperation::kSUB},            // 4
      {c10::aten::div, nvinfer1::ElementWiseOperation::kDIV},             // 5
      {c10::aten::div_, nvinfer1::ElementWiseOperation::kDIV},            // 5
      {c10::aten::pow, nvinfer1::ElementWiseOperation::kPOW},             // 6
      {c10::aten::floordiv, nvinfer1::ElementWiseOperation::kFLOOR_DIV},  // 7
      {c10::aten::eq, nvinfer1::ElementWiseOperation::kEQUAL},            // 11
      {c10::aten::gt, nvinfer1::ElementWiseOperation::kGREATER},          // 12
      {c10::aten::lt, nvinfer1::ElementWiseOperation::kLESS},             // 13
  };
};

FWD_TORCH_NAMESPACE_END
