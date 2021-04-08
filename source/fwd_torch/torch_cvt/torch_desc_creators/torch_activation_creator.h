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

// Activation Description Creator
template <>
class TLayerDescCreator<TrtActivationDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return NK2AT_MAPPING.find(node->kind()) != NK2AT_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtActivationDesc::Create";

    auto layer_desc = std::make_shared<TrtActivationDesc>();

    const auto inputs = node->inputs();

    // Input 0, kind = input
    // Input 1 (optional) kind = c10::prim::Constant alpha
    // Input 2 (optional) kind = c10::prim::Constant beta
    input_values.push_back(inputs[0]);

    const auto kind = node->kind();
    layer_desc->activationType = NK2AT_MAPPING.find(kind)->second;

    if (inputs.size() > 1) layer_desc->alpha = module.Get(inputs[1]).toScalar().toFloat();
    if (inputs.size() > 2) layer_desc->beta = module.Get(inputs[2]).toScalar().toFloat();

    return layer_desc;
  }

 private:
#define CSYM(s) c10::Symbol::fromQualString(#s)
  const std::unordered_map<c10::Symbol, nvinfer1::ActivationType> NK2AT_MAPPING = {
      {c10::aten::relu, nvinfer1::ActivationType::kRELU},                // 0
      {CSYM(aten::relu_), nvinfer1::ActivationType::kRELU},              // 0
      {c10::aten::sigmoid, nvinfer1::ActivationType::kSIGMOID},          // 1
      {c10::aten::tanh, nvinfer1::ActivationType::kTANH},                // 2
      {c10::aten::leaky_relu, nvinfer1::ActivationType::kLEAKY_RELU},    // 3
      {CSYM(aten::leaky_relu_), nvinfer1::ActivationType::kLEAKY_RELU},  // 3
      {c10::aten::elu, nvinfer1::ActivationType::kELU},                  // 4
      {c10::aten::selu, nvinfer1::ActivationType::kSELU},                // 5
      {c10::aten::softplus, nvinfer1::ActivationType::kSOFTPLUS},        // 7
  };
#undef CSYM
};

FWD_TORCH_NAMESPACE_END
