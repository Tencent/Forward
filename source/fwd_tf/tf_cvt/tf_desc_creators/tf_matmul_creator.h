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
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief Padding 层描述创建器
 */
template <>
class TLayerDescCreator<TrtMatrixMultiplyDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    return type == "MatMul";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtMatMulDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 2);

    const auto input0 = op.Input(0);
    const auto input1 = op.Input(1);

    const auto const_tensor_0 = ToFwdWeights(input0.GetConstantTensor());
    const auto const_tensor_1 = ToFwdWeights(input1.GetConstantTensor());

    auto layer_desc = std::make_shared<TrtMatrixMultiplyDesc>();

    Status status;

    if (!const_tensor_0.Empty()) {
      layer_desc->inputs[0].inUse = true;
      layer_desc->inputs[0].data = const_tensor_0;
      layer_desc->inputs[0].dim = DimsOf(input0);
    } else {
      op_inputs.push_back(input0);
    }

    if (!const_tensor_1.Empty()) {
      layer_desc->inputs[1].inUse = true;
      layer_desc->inputs[1].data = const_tensor_1;
      layer_desc->inputs[1].dim = DimsOf(input1);
    } else {
      op_inputs.push_back(input1);
    }

    auto transpose_a = op.GetAttrBool("transpose_a");
    auto transpose_b = op.GetAttrBool("transpose_b");

    layer_desc->op0 = static_cast<nvinfer1::MatrixOperation>(transpose_a);
    layer_desc->op1 = static_cast<nvinfer1::MatrixOperation>(transpose_b);

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
