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
#include <utility>
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief Constant 层描述创建器
 */
template <>
class TLayerDescCreator<TrtConstantDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    // TODO(zhaoyiluo): tf.constant
    const auto type = op.OpType();
    return type == "Fill";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtConstantDesc::Create";

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 2);

    op_inputs.push_back({});

    // Input 0, dims
    // Input 1, value
    const auto input0 = op.Input(0);
    const auto input1 = op.Input(1);

    if (input0.GetTensorNumDims() != 1) {
      LOG(ERROR) << "dims of tf.fill should be a 1-D sequence of non-negative numbers, at "
                 << input0.Name();
      return nullptr;
    }

    const auto value_tensor = input1.GetConstantTensor();
    if (value_tensor.ElementCount() != 1) {
      LOG(ERROR) << "value of tf.fill should be a scalar, at " << input1.Name();
    }

    const auto dims = input0.GetConstantTensor().AsIntList();
    const auto tensor = CreateWeightTensor(dims, value_tensor);  // create float-tensor only

    auto layer_desc = std::make_shared<TrtConstantDesc>();
    layer_desc->weights = ToFwdWeights(tensor.get());
    layer_desc->dimensions = TrtUtils::ToDims(dims);

    return layer_desc;
  }

 private:
  std::shared_ptr<TF_Tensor> CreateWeightTensor(const std::vector<int>& dims,
                                                const Tensor& tensor) {
    float value;
    {
      switch (tensor.Type()) {
        case TF_INT8:
          value = static_cast<float>(tensor.Data<int8_t>()[0]);
          break;
        case TF_INT16:
          value = static_cast<float>(tensor.Data<int16_t>()[0]);
          break;
        case TF_INT32:
          value = static_cast<float>(tensor.Data<int32_t>()[0]);
          break;
        default:
          value = static_cast<float>(tensor.Data<float>()[0]);
          break;
      }
    }

    std::vector<int64_t> shape;
    shape.reserve(dims.size());

    for (auto& dim : dims) {
      shape.emplace_back(std::move(dim));
    }

    return CreateConstantTensor(TF_FLOAT, shape, value);
  }
};

FWD_TF_NAMESPACE_END
