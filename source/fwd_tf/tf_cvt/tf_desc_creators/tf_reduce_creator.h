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
/**
 * \brief Reduce 层描述创建器
 */
template <>
class TLayerDescCreator<TrtReduceDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    return OT2RO_MAPPING.find(type) != OT2RO_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtReduceDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const int num_inputs = op.NumInputs();

    T_CHECK_EQ(num_inputs, 2);

    const std::string type = op.OpType();

    const auto input = op.Input(0);
    const auto reduce_indices = op.Input(1);
    const bool keep_dims = op.GetAttrBool("keep_dims");

    // Status status;
    const auto input_nbDims = input.GetTensorNumDims();

    op_inputs.push_back(input);

    auto layer_desc = std::make_shared<TrtReduceDesc>();

    layer_desc->keepDimensions = keep_dims;
    layer_desc->operation = OT2RO_MAPPING.find(type)->second;

    const auto reduce_dims = reduce_indices.GetConstantTensor().AsIntList();

    for (auto dim : reduce_dims) {
      if (input_nbDims == 4) {
        layer_desc->reduceAxes |= (1 << TrtUtils::NHWC2NCHWDim(dim));
      } else {
        layer_desc->reduceAxes |= (1 << dim);
      }
    }

    return layer_desc;
  }

 private:
  const std::unordered_map<std::string, nvinfer1::ReduceOperation> OT2RO_MAPPING = {
      {"Mean", nvinfer1::ReduceOperation::kAVG},
      {"Sum", nvinfer1::ReduceOperation::kSUM},
      {"Max", nvinfer1::ReduceOperation::kMAX},
      {"Min", nvinfer1::ReduceOperation::kMIN},
  };
};

FWD_TF_NAMESPACE_END
