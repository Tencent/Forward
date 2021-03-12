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
 * \brief Concatenation 层描述创建器
 */
template <>
class TLayerDescCreator<TrtConcatenationDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();

    return type == "ConcatV2";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtConcatenationDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const auto num_inputs = op.NumInputs();

    // 获取属性
    int64_t N = op.GetAttrInt("N");
    T_CHECK_EQ(num_inputs, N + 1);

    // 将输入返回
    for (int i = 0; i < num_inputs - 1; ++i) {
      op_inputs.push_back(op.Input(i));
    }

    auto layer_desc = std::make_shared<TrtConcatenationDesc>();

    const int axis = op.Input(num_inputs - 1).GetConstantTensor().AsInt();
    // const int axis = Utils::GetConstantInt(graph, op.Input(num_inputs - 1));

    auto input0 = op.Input(0);
    const auto nbDims = input0.GetTensorNumDims();

    // tensorflow NHWC -> TensorRT NCHW
    layer_desc->axis = (nbDims == 4 ? TrtUtils::NHWC2NCHWDim(axis) : axis);

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
