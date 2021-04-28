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

FWD_TF_NAMESPACE_BEGIN
// ElementWise Description Creator
template <>
class TLayerDescCreator<TrtElementWiseDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const std::string type = op.OpType();

    return NK2EWP_MAPPING.find(type) != NK2EWP_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtElementWiseDesc::Create";

    const int num_inputs = op.NumInputs();

    const std::string operation_type = op.OpType();

    if (operation_type == "Square") {
      return CreateSquareOp(op, op_inputs, operation_type);
    }

    return CreateNormalOp(op, op_inputs, num_inputs, operation_type);
  }

 private:
  std::shared_ptr<TrtLayerDesc> CreateSquareOp(const Operation& op, std::vector<Output>& op_inputs,
                                               const std::string operation_type) const {
    auto layer_desc = std::make_shared<TrtElementWiseDesc>();

    const auto input0 = op.Input(0);
    op_inputs.push_back(input0);
    op_inputs.push_back(input0);

    layer_desc->operation = NK2EWP_MAPPING.find(operation_type)->second;
    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateNormalOp(const Operation& op, std::vector<Output>& op_inputs,
                                               const int num_inputs,
                                               const std::string operation_type) const {
    auto layer_desc = std::make_shared<TrtElementWiseDesc>();
    layer_desc->operation = NK2EWP_MAPPING.find(operation_type)->second;

    for (int i = 0; i < num_inputs; ++i) {
      const auto input = op.Input(i);
      const auto tensor = ToFwdWeights(input.GetConstantTensor());

      if (tensor.Empty()) {
        op_inputs.push_back(input);
      } else {
        layer_desc->inputs[i].inUse = true;
        layer_desc->inputs[i].data = tensor;
        layer_desc->inputs[i].dim = DimsOf(input);
        op_inputs.push_back(Output());
      }
    }

    return layer_desc;
  }

  const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> NK2EWP_MAPPING = {
      {"Add", nvinfer1::ElementWiseOperation::kSUM},      // 0
      {"AddV2", nvinfer1::ElementWiseOperation::kSUM},    // 0
      {"Mul", nvinfer1::ElementWiseOperation::kPROD},     // 1
      {"Sub", nvinfer1::ElementWiseOperation::kSUB},      // 4
      {"RealDiv", nvinfer1::ElementWiseOperation::kDIV},  // 5
      {"Pow", nvinfer1::ElementWiseOperation::kPOW},      // 6
      {"Square", nvinfer1::ElementWiseOperation::kPROD},  // 1
#if NV_TENSORRT_MAJOR >= 7
      {"Greater", nvinfer1::ElementWiseOperation::kGREATER},  // 12
      {"Less", nvinfer1::ElementWiseOperation::kLESS},        // 13
#endif // NV_TENSORRT_MAJOR >= 7
  };
};

FWD_TF_NAMESPACE_END
