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

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief Activation 激活层描述创建器
 */
template <>
class TLayerDescCreator<TrtActivationDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();
    return NK2AT_MAPPING.find(type) != NK2AT_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtActivationDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 1);

    // Input 0, kind = input
    const auto input = op.Input(0);

    // 将输入返回
    op_inputs.push_back(input);

    auto layer_desc = std::make_shared<TrtActivationDesc>();

    const auto type = op.OpType();
    layer_desc->activationType = NK2AT_MAPPING.find(type)->second;

    return layer_desc;
  }

 private:
  const std::unordered_map<std::string, nvinfer1::ActivationType> NK2AT_MAPPING = {
      {"Relu", nvinfer1::ActivationType::kRELU},        // 0
      {"Sigmoid", nvinfer1::ActivationType::kSIGMOID},  // 1
      {"Tanh", nvinfer1::ActivationType::kTANH},        // 2
      {"Elu", nvinfer1::ActivationType::kELU},          // 3
  };
};

FWD_TF_NAMESPACE_END
