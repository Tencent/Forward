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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief TopK 层描述创建器
 */
template <>
class TLayerDescCreator<TrtTopKDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();
    return type == "TopKV2";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtTopKDesc::Create";

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 2);

    const auto input0 = op.Input(0);  // input
    const auto input1 = op.Input(1);  // scalar K

    const int num_dims = input0.GetTensorNumDims();
    if (num_dims == 0) {
      LOG(ERROR) << "TensorRT TopK cannot apply on batch dimension, at " << input0.Name();
      return nullptr;
    }

    const auto scalar = input1.GetConstantTensor();
    if (scalar.ElementCount() != 1) {
      LOG(ERROR) << "k value of TopK should be a scalar, at " << input1.Name();
    }

    if (!op.GetAttrBool("sorted")) {
      LOG(ERROR) << "TensorRT only supports sorted output";
      return nullptr;
    }

    op_inputs.push_back(input0);

    auto layer_desc = std::make_shared<TrtTopKDesc>();
    layer_desc->operation = nvinfer1::TopKOperation::kMAX;  // with no alternatives by default
    layer_desc->k = scalar.AsInt();
    layer_desc->reduceAxes = 1 << (num_dims - 1);

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
