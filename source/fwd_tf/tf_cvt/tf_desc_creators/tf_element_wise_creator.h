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
#include "fwd_tf/tf_cvt/tf_utils.h"

#define TensorRT7

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief ElementWise 层描述创建器
 */
template <>
class TLayerDescCreator<TrtElementWiseDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    return NK2EWP_MAPPING.find(type) != NK2EWP_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtElementWiseDesc::Create";

    const int num_inputs = op.NumInputs();

    const std::string operation_type = op.OpType();

    if (operation_type == "Square") {
      auto layer_desc = std::make_shared<TrtElementWiseDesc>();

      // const auto input0 = TF_OperationInput({ op, 0 });
      const auto input0 = op.Input(0);
      op_inputs.push_back(input0);
      op_inputs.push_back(input0);

      layer_desc->operation = NK2EWP_MAPPING.find(operation_type)->second;
      return layer_desc;
    }

    auto layer_desc = std::make_shared<TrtElementWiseDesc>();
    layer_desc->operation = NK2EWP_MAPPING.find(operation_type)->second;

    // const auto input0 = TF_OperationInput({ op, 0 });
    const auto input0 = op.Input(0);
    const auto const_tensor_0 = Utils::ToFwdWeights(input0.GetConstantTensor());

    // input1 = TF_OperationInput({ op, 1 });
    const auto input1 = op.Input(1);
    const auto const_tensor_1 = Utils::ToFwdWeights(input1.GetConstantTensor());

    // 输入一方或两方可能为常量 Tensor 的情况
    if (const_tensor_0.Empty()) {
      op_inputs.push_back(input0);
    } else {
      layer_desc->inputs[0].inUse = true;
      layer_desc->inputs[0].data = const_tensor_0;
      layer_desc->inputs[0].dim = Utils::DimsOf(input0);
    }

    if (const_tensor_1.Empty()) {
      op_inputs.push_back(input1);
    } else {
      layer_desc->inputs[1].inUse = true;
      layer_desc->inputs[1].data = const_tensor_1;
      layer_desc->inputs[1].dim = Utils::DimsOf(input1);
    }

    if (op_inputs.empty()) {
      op_inputs.push_back(Output());
    }

    return layer_desc;
  }

 private:
  const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> NK2EWP_MAPPING = {
      {"Add", nvinfer1::ElementWiseOperation::kSUM},      // 0
      {"AddV2", nvinfer1::ElementWiseOperation::kSUM},    // 0
      {"Mul", nvinfer1::ElementWiseOperation::kPROD},     // 1
      {"Sub", nvinfer1::ElementWiseOperation::kSUB},      // 4
      {"RealDiv", nvinfer1::ElementWiseOperation::kDIV},  // 5
      {"Pow", nvinfer1::ElementWiseOperation::kPOW},      // 6
      {"Square", nvinfer1::ElementWiseOperation::kPROD},  // 1
#ifdef TensorRT7
      {"Greater", nvinfer1::ElementWiseOperation::kGREATER},  // 12
      {"Less", nvinfer1::ElementWiseOperation::kLESS},        // 13
#endif                                                        // ifdef TensorRT7
  };
};

FWD_TF_NAMESPACE_END
