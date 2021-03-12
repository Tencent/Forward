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
 * \brief embedding_bag/gather 层描述创建器
 */
template <>
class TLayerDescCreator<TrtEmbeddingBagDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    if (type == "GatherV2") return true;

    if (type != "Sum") return false;

    auto father = op;
    father = father.Input(0);
    type = father.OpType();

    if (type == "Square") {
      father = father.Input(0);
      type = father.OpType();
    }

    if (type != "Mul") return false;
    father = father.Input(0);
    type = father.OpType();

    if (type == "Identity") {
      father = father.Input(0);
      type = father.OpType();
    }

    return type == "GatherV2";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtEmbeddingBagDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    int op_type = 0;

    auto father = op;  // Sum Node or Gather Node

    if (father.OpType() == "Sum") {
      father = father.Input(0);
      std::string type = father.OpType();
      if (type == "Square") {
        father = father.Input(0);  // Mul Node
        type = father.OpType();
        op_type = 3;
      }
      father = father.Input(0);  // Identity/GatherV2 Node
      type = father.OpType();
      if (type == "Identity") {
        father = father.Input(0);  // GatherV2 Node
      }
    } else {
      op_type = 999;
    }

    const auto num_inputs = father.NumInputs();
    T_CHECK_EQ(num_inputs, 3);

    // Input 0, kind = input
    // Input 1, indices
    // Input 2, axis
    const auto input = father.Input(0);
    const auto indices = father.Input(1);
    const auto gather_axis = father.Input(2).GetConstantTensor().AsInt();
    T_CHECK_EQ(gather_axis, 0);

    // 先查找是否有根据文件生成的对应权重Op
    std::string input_name = input.Name();
    input_name = input_name.substr(0, input_name.find_last_of('/')) + "_weights";

    FwdWeights const_tensor_0;
    if (graph.OperationByName(input_name) != nullptr) {
      const auto input_from_file = Output(graph.get(), graph.OperationByName(input_name), 0);
      const_tensor_0 = Utils::ToFwdWeights(input_from_file.GetConstantTensor());
    } else {
      const_tensor_0 = Utils::ToFwdWeights(input.GetConstantTensor());
    }
    const auto const_tensor_1 = Utils::ToFwdWeights(indices.GetConstantTensor());

    auto layer_desc = std::make_shared<TrtEmbeddingBagDesc>();

    const std::shared_ptr<TF_Status> status(TF_NewStatus(), TF_DeleteStatus);

    if (!const_tensor_0.Empty()) {
      layer_desc->data = const_tensor_0;
    } else {
      T_CHECK(false);
    }
    auto dims = Utils::DimsOf(input);
    layer_desc->count = dims.d[0];
    layer_desc->dim = dims.nbDims == 1 ? 1 : dims.d[1];
    layer_desc->offset = 0;
    layer_desc->op = op_type;

    op_inputs.push_back(indices);

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
