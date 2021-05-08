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
 * \brief Unary 层描述创建器
 */
template <>
class TLayerDescCreator<TrtUnaryDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();
    return NK2UO_MAPPING.find(kind) != NK2UO_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtUnaryDesc::Create";

    const auto inputs = node->inputs();

    auto layer_desc = std::make_shared<TrtUnaryDesc>();

    // Input 0, kind = input
    if (inputs[0]->node()->kind() == c10::prim::GetAttr) {
      const auto input_tensor = module.Get(inputs[0]).toTensor();
      input_values.push_back(nullptr);
      layer_desc->input.inUse = true;
      layer_desc->input.data = ToFwdWeights(input_tensor);
      layer_desc->input.dim = DimsOf(input_tensor);
    } else {
      input_values.push_back(inputs[0]);
    }

    const auto kind = node->kind();
    layer_desc->operation = NK2UO_MAPPING.find(kind)->second;

    return layer_desc;
  }

 private:
#define CSYM(s) c10::Symbol::fromQualString(#s)

  const std::unordered_map<c10::Symbol, nvinfer1::UnaryOperation> NK2UO_MAPPING = {
    {c10::aten::exp, nvinfer1::UnaryOperation::kEXP},             // 0
    {CSYM(aten::exp_), nvinfer1::UnaryOperation::kEXP},           // 0
    {c10::aten::log, nvinfer1::UnaryOperation::kLOG},             // 1
    {CSYM(aten::log_), nvinfer1::UnaryOperation::kLOG},           // 1
    {c10::aten::sqrt, nvinfer1::UnaryOperation::kSQRT},           // 2
    {CSYM(aten::sqrt_), nvinfer1::UnaryOperation::kSQRT},         // 2
    {c10::aten::reciprocal, nvinfer1::UnaryOperation::kRECIP},    // 3
    {CSYM(aten::reciprocal_), nvinfer1::UnaryOperation::kRECIP},  // 3
    {c10::aten::abs, nvinfer1::UnaryOperation::kABS},             // 4
    {CSYM(aten::abs_), nvinfer1::UnaryOperation::kABS},           // 4
    {c10::aten::neg, nvinfer1::UnaryOperation::kNEG},             // 5
    {CSYM(aten::neg_), nvinfer1::UnaryOperation::kNEG},           // 5
    {c10::aten::sin, nvinfer1::UnaryOperation::kSIN},             // 6
    {CSYM(aten::sin_), nvinfer1::UnaryOperation::kSIN},           // 6
    {c10::aten::cos, nvinfer1::UnaryOperation::kCOS},             // 7
    {CSYM(aten::cos_), nvinfer1::UnaryOperation::kCOS},           // 7
    {c10::aten::tan, nvinfer1::UnaryOperation::kTAN},             // 8
    {CSYM(aten::tan_), nvinfer1::UnaryOperation::kTAN},           // 8
    {c10::aten::sinh, nvinfer1::UnaryOperation::kSINH},           // 9
    {CSYM(aten::sinh_), nvinfer1::UnaryOperation::kSINH},         // 9
    {c10::aten::cosh, nvinfer1::UnaryOperation::kCOSH},           // 10
    {CSYM(aten::cosh_), nvinfer1::UnaryOperation::kCOSH},         // 10
    {c10::aten::asin, nvinfer1::UnaryOperation::kASIN},           // 11
    {CSYM(aten::asin_), nvinfer1::UnaryOperation::kASIN},         // 11
    {c10::aten::acos, nvinfer1::UnaryOperation::kACOS},           // 12
    {CSYM(aten::acos_), nvinfer1::UnaryOperation::kACOS},         // 12
    {c10::aten::atan, nvinfer1::UnaryOperation::kATAN},           // 13
    {CSYM(aten::atan_), nvinfer1::UnaryOperation::kATAN},         // 13
    {CSYM(aten::asinh), nvinfer1::UnaryOperation::kASINH},        // 14
    {CSYM(aten::asinh_), nvinfer1::UnaryOperation::kASINH},       // 14
    {CSYM(aten::acosh), nvinfer1::UnaryOperation::kACOSH},        // 15
    {CSYM(aten::acosh_), nvinfer1::UnaryOperation::kACOSH},       // 15
    {CSYM(aten::atanh), nvinfer1::UnaryOperation::kATANH},        // 16
    {CSYM(aten::atanh_), nvinfer1::UnaryOperation::kATANH},       // 16
    {c10::aten::ceil, nvinfer1::UnaryOperation::kCEIL},           // 17
    {CSYM(aten::ceil_), nvinfer1::UnaryOperation::kCEIL},         // 17
    {c10::aten::floor, nvinfer1::UnaryOperation::kFLOOR},         // 18
    {CSYM(aten::floor_), nvinfer1::UnaryOperation::kFLOOR},       // 18
#if NV_TENSORRT_MAJOR >= 7
    {CSYM(aten::erf), nvinfer1::UnaryOperation::kERF},   // 19
    {CSYM(aten::erf_), nvinfer1::UnaryOperation::kERF},  // 19
#endif                                                   // NV_TENSORRT_MAJOR >= 7
  };

#undef CSYM
};

FWD_TORCH_NAMESPACE_END
