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
 * \brief FullyConnected 全连接层描述创建器
 */
template <>
class TLayerDescCreator<TrtFullyConnectedDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    if (type != "BiasAdd") return false;

    const auto input = op.Input(0);
    type = input.OpType();

    if (type != "MatMul") return false;

    return true;
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtFullyConnectedDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 2);

    // Input 0, kind = input
    const auto bias_input = op.Input(0);
    const auto bias = op.Input(1);

    const auto mat_mul_op = bias_input;
    const auto mat_mul_input = mat_mul_op.Input(0);
    const auto weights = mat_mul_op.Input(1);

    // 将输入返回
    op_inputs.push_back(mat_mul_input);

    Status status;
    uint8_t transpose_a;
    uint8_t transpose_b;
    TF_OperationGetAttrBool(mat_mul_op.Op(), "transpose_a", &transpose_a, status);
    TF_OperationGetAttrBool(mat_mul_op.Op(), "transpose_b", &transpose_b, status);

    auto layer_desc = std::make_shared<TrtFullyConnectedDesc>();

    // TODO(Paul Lu): fc层未遇到过此处为true的情形; 如果出现了再做处理。
    T_CHECK(!transpose_a);

    layer_desc->kernelWeights = ToFwdWeights(weights.GetConstantTensor());
    const auto dims = DimsOf(weights);
    if (!transpose_b) {
      layer_desc->kernelWeights.Transpose(dims, {1, 0});
    }

    layer_desc->biasWeights = ToFwdWeights(bias.GetConstantTensor());

    layer_desc->nbOutputChannels = layer_desc->biasWeights.Count();

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
