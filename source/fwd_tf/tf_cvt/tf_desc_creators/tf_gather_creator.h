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
class TLayerDescCreator<TrtGatherDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    return type == "GatherV2";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtGatherDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 3);

    // Input 0, kind = input
    // Input 1, indices
    // Input 2, axis
    const auto input = op.Input(0);
    const auto indices = op.Input(1);

    std::string input_name = input.Name();
    input_name = input_name.substr(0, input_name.find_last_of('/')) + "_weights";

    FwdWeights const_tensor_0;
    if (graph.OperationByName(input_name) != nullptr) {
      const auto input_from_file = Output(graph.get(), graph.OperationByName(input_name), 0);
      const_tensor_0 = ToFwdWeights(input_from_file.GetConstantTensor());
    } else {
      const_tensor_0 = ToFwdWeights(input.GetConstantTensor());
    }
    const auto const_tensor_1 = ToFwdWeights(indices.GetConstantTensor());

    auto layer_desc = std::make_shared<TrtGatherDesc>();

    Status status;

    if (!const_tensor_0.Empty()) {
      layer_desc->inputs[0].inUse = true;
      layer_desc->inputs[0].data = const_tensor_0;
      layer_desc->inputs[0].dim = DimsOf(input);
    } else {
      op_inputs.push_back(input);
    }

    if (!const_tensor_1.Empty()) {
      layer_desc->inputs[1].inUse = true;
      layer_desc->inputs[1].data = const_tensor_1;
      layer_desc->inputs[1].dim = DimsOf(indices);
    } else {
      op_inputs.push_back(indices);
    }

    layer_desc->gatherAxis = op.Input(2).GetConstantTensor().AsInt();

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
