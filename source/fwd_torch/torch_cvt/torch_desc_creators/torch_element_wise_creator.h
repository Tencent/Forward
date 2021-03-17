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

/**
 * \brief ElementWise 层描述创建器
 */
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
    T_CHECK_GE(inputs.size(), 2);
    // Input 0: Tensor input0
    // Input 1: Tensor input1
    // Input 2: Scalar alpha, add, add_, sub, rsub

    auto layer_desc = std::make_shared<TrtElementWiseDesc>();

    const auto kind = node->kind();

    // op = rsub 读入顺序为 inputs[1], inputs[0]; 否则为inputs[0], inputs[1]
    const bool isRSub = (kind == c10::aten::rsub);

    // 两个输入都可能从NumToTensor()/常数得到，若是则需要单独储存
    for (int i = 0; i < 2; i++) {
      if (inputs[i ^ isRSub]->node()->kind() == c10::prim::GetAttr) {
        layer_desc->inputs[i].inUse = true;
        T_CHECK(module.Get(inputs[i ^ isRSub]).isTensor());
        auto tensor = module.Get(inputs[i ^ isRSub]).toTensor();
        layer_desc->inputs[i].data = ToFwdWeights(tensor);
        layer_desc->inputs[i].dim = DimsOf(tensor);
      } else if (inputs[i ^ isRSub]->node()->kind() == c10::prim::Constant) {
        layer_desc->inputs[i].inUse = true;
        const auto& input = module.Get(inputs[i ^ isRSub]);
        if (input.isTensor()) {
          layer_desc->inputs[i].data = ToFwdWeights(input.toTensor());
          layer_desc->inputs[i].dim = DimsOf(input.toTensor());
        } else if (input.isInt()) {
          layer_desc->inputs[i].data = ToFwdWeights(input.toInt());
          layer_desc->inputs[i].dim = {1, 1};
        } else if (input.isDouble()) {
          layer_desc->inputs[i].data = ToFwdWeights(input.toDouble());
          layer_desc->inputs[i].dim = {1, 1};
        } else {
          LOG(ERROR) << "Input " << i << "'s Type " << input.type()->str() << " Not Supported yet.";
          LOG(ERROR) << "Create Desc Failed.";
          T_CHECK(false);
        }
      } else {
        input_values.push_back(inputs[i ^ isRSub]);
      }
    }

    // Tensor op Scalar 时，可能会出现alpha != 1, 系数与Scalar相乘
    int alpha = inputs.size() > 2 ? module.Get(inputs[2]).toInt() : 1;
    if (alpha != 1) {
      float value = *(reinterpret_cast<const float*>(layer_desc->inputs[1 ^ isRSub].data.Data()));
      layer_desc->inputs[1 ^ isRSub].data = torch_::ToFwdWeights(value * alpha);
    }

    layer_desc->operation = NK2EWP_MAPPING.find(kind)->second;

    return layer_desc;
  }

 private:
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
