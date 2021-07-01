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
#include <unordered_map>
#include <utility>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Cast 层描述器
 */
template <>
class TLayerDescCreator<TrtCastDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::to && node->schema().overload_name() == "dtype";
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    const auto inputs = node->inputs();
    T_CHECK_GE(inputs.size(), 2);

    const auto input = inputs[0];

    // if the input is from a prim::GetAttr, which means Constant Weights, then we create a constant
    // layer for a constant weight.
    if (input->node()->kind() == c10::prim::GetAttr) {
      input_values.push_back(nullptr);
      auto layer_desc = std::make_shared<TrtConstantDesc>();
      auto ivalue = module.Get(node->output());
      T_CHECK(ivalue.isTensor());
      auto tensor = ivalue.toTensor();
      layer_desc->weights = ToFwdWeights(tensor);
      layer_desc->dimensions = layer_desc->weights.Dims();
      return layer_desc;
    }

    input_values.push_back(input);

    auto layer_desc = std::make_shared<TrtCastDesc>();
    const auto type = module.Get(inputs[1]).toScalarType();
    layer_desc->otype = ST2DT_MAPPING.at(c10::toString(type));

    return layer_desc;
  }

 private:
  const std::unordered_map<const char*, nvinfer1::DataType> ST2DT_MAPPING = {
    {c10::toString(c10::ScalarType::Float), nvinfer1::DataType::kFLOAT},
    {c10::toString(c10::ScalarType::Half), nvinfer1::DataType::kHALF},
    {c10::toString(c10::ScalarType::QInt8), nvinfer1::DataType::kINT8},
    {c10::toString(c10::ScalarType::Int), nvinfer1::DataType::kINT32},
#if NV_TENSORRT_MAJOR >= 7
    {c10::toString(c10::ScalarType::Bool), nvinfer1::DataType::kBOOL},
#endif  // NV_TENSORRT_MAJOR >= 7
  };
};

FWD_TORCH_NAMESPACE_END
