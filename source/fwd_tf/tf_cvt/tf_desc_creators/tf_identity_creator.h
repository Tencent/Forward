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
class TLayerDescCreator<TrtIdentityDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();
    const TF_DataType dtype = op.InputType(0);
    const TF_DataType o_dtype = op.OutputType(0);

    return type == "Identity" || type == "Range" || CheckMaskPedding(op) ||
           (type == "Cast" && (dtype != TF_DataType::TF_BOOL || o_dtype == TF_DataType::TF_BOOL));
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtIdentityDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    std::string type = op.OpType();

    if (type == "Range") {
      return CreateRange(op, graph, op_inputs);
    }

    // 忽略MaskPedding
    if (CheckMaskPedding(op)) {
      return CreateIdentity(op, graph, op_inputs, true);
    }

    if (type == "Cast") {
      return CreateIdentity(op, graph, op_inputs, false);
    }

    // If type == "Identity"

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 1);

    Status status;

    // Input 0, kind = input
    const auto input = op.Input(0);

    auto const_tensor = ToFwdWeights(input.GetConstantTensor());

    if (const_tensor.Empty()) {
      auto layer_desc = std::make_shared<TrtIdentityDesc>();
      op_inputs.push_back(input);
      return layer_desc;
    }

    auto layer_desc = std::make_shared<TrtIdentityDesc>();

    layer_desc->input.inUse = true;
    layer_desc->input.data = const_tensor;
    layer_desc->input.dim = DimsOf(input);

    op_inputs.push_back(Output());

    return layer_desc;
  }

 private:
  std::shared_ptr<TrtLayerDesc> CreateRange(const Operation& op, const Graph& graph,
                                            std::vector<Output>& op_inputs) {
    LOG(INFO) << "TrtIdentityDesc::CreateRange";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    std::string type = op.OpType();

    const auto start_op = op.Input(0);
    const auto end_op = op.Input(1);
    const auto delta_op = op.Input(2);

    int start = 0;
    int end = 0;
    int delta = 0;

    auto layer_desc = std::make_shared<TrtIdentityDesc>();
    layer_desc->input.inUse = true;
    layer_desc->input.dim.nbDims = 0;

    auto tensor = start_op.GetConstantTensor();
    if (tensor.Valid()) {
      start = tensor.AsInt();
    } else {
      if (!FindShapePattern(graph, start_op, op_inputs, start, layer_desc->input.dim)) {
        LOG(ERROR) << "Operation Currently Not Supported";
        T_CHECK(false);
      }
    }

    auto end_op_tensor = end_op.GetConstantTensor();
    if (end_op_tensor.Valid()) {
      end = end_op_tensor.AsInt();
    } else {
      if (!FindShapePattern(graph, end_op, op_inputs, end, layer_desc->input.dim)) {
        LOG(ERROR) << "Operation Currently Not Supported";
        T_CHECK(false);
      }
    }

    auto delta_tensor = delta_op.GetConstantTensor();
    if (delta_tensor.Valid()) {
      delta = delta_tensor.AsInt();
    } else {
      if (!FindShapePattern(graph, delta_op, op_inputs, delta, layer_desc->input.dim)) {
        LOG(ERROR) << "Operation Currently Not Supported";
        T_CHECK(false);
      }
    }

    std::vector<int> range((end - start - 1) / delta + 1);
    range[0] = start;
    for (int i = 1; i < range.size(); i++) {
      range[i] = range[i - 1] + delta;
    }
    layer_desc->input.data = FwdWeights(range);

    // 三个参数全部是常数，此时应该会展开成一个常量Tensor，预计不会触发
    if (layer_desc->input.dim.nbDims == 0) {
      layer_desc->input.dim.nbDims = 1;
      layer_desc->input.dim.d[0] = range.size();
    } else {
      // Todo(yzx):
      // 这里的处理实际上只针对参数来源于原始的第一/第二维有效，需考虑是否可能有更多情况
      // 如果创造一个1维tensor，在element_wise操作中可能无法正确地broadcast
      layer_desc->input.dim.nbDims = 2;
      for (int i = 0; i < layer_desc->input.dim.nbDims; i++) {
        if (layer_desc->input.dim.d[i] != 1) layer_desc->input.dim.d[i] = range.size();
      }
    }

    op_inputs.push_back(Output());

    return layer_desc;
  }

  static bool FindShapePattern(const Graph& graph, const Operation& oper,
                               std::vector<Output>& op_inputs, int& dim_value,
                               nvinfer1::Dims& dims) {
    std::string type = oper.OpType();
    if (type != "StridedSlice") {
      return false;
    }

    // find which dim info to use
    int d = oper.Input(1).GetConstantTensor().AsInt();
    // int d = GetConstantInt(graph, TF_OperationInput(TF_Input{ oper, 1
    // }));

    Output op = oper.Input(0);
    // Operation op = Operation(graph.get(), TF_OperationInput(TF_Input{ op,0
    // }).oper);
    type = op.OpType();
    if (type != "Shape") {
      return false;
    }

    op = op.Input(0);
    auto shape = DimsOf(op);
    dim_value = shape.d[d];

    dims = shape;
    for (int i = 0; i < dims.nbDims; i++) {
      if (i != d) dims.d[i] = 1;
    }

    return true;
  }

  bool CheckMaskPedding(const Operation& op) {
    std::string type = op.OpType();

    if (type != "Mul") return false;

    auto father = op;

    // 支线1
    std::vector<std::string> route1{"Cast", "Less", "Range", "StridedSlice", "Shape"};
    for (int i = 0; i < 5; i++) {
      // Mul和Range层是第二个参数
      // const auto input = TF_OperationInput(TF_Input{ father, (i == 0) + (i ==
      // 3) });
      const auto input = father.Input((i == 0) + (i == 3));
      father = input;
      std::string type = father.OpType();

      if (type != route1[i]) {
        return false;
      }

      // 中间可能有ExpandDims
      if (i == 0) {
        // std::string grandfather_type =
        // TF_OperationOpType(TF_OperationInput(TF_Input{ father, 0 }).oper);
        std::string grandfather_type = father.Input(0).OpType();
        if (grandfather_type == "ExpandDims") {
          // father = TF_OperationInput(TF_Input{ father, 0 }).oper;
          father = father.Input(0);
        }
      }
    }
    if (father.Input(0).Op() != op.Input(0).Op()) {
      return false;
    }

    // 支线2
    father = op;
    std::vector<std::string> route2{"Cast", "Less", "Cast", "ExpandDims", "Reshape"};
    for (int i = 0; i < 5; i++) {
      // Mul和Range层是第二个参数
      // const auto input = TF_OperationInput(TF_Input{ father, (i == 0) + (i ==
      // 2) }); father = input.oper;
      //
      const auto input = father.Input((i == 0) + (i == 2));
      father = input;

      const auto type = father.OpType();
      if (type != route2[i]) {
        return false;
      }

      // 中间可能有ExpandDims
      if (i == 0) {
        // std::string grandfather_type =
        // TF_OperationOpType(TF_OperationInput(TF_Input{ father, 0 }).oper);
        std::string grandfather_type = father.Input(0).OpType();
        if (grandfather_type == "ExpandDims") {
          // father = TF_OperationInput(TF_Input{ father, 0 }).oper;
          father = father.Input(0);
        }
      }
    }
    return true;
  }

  std::shared_ptr<TrtLayerDesc> CreateIdentity(const Operation& op, const Graph& graph,
                                               std::vector<Output>& op_inputs, bool do_copy) {
    const auto input = op.Input(0);

    auto layer_desc = std::make_shared<TrtIdentityDesc>();
    op_inputs.push_back(input);
    layer_desc->copy = do_copy;
    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
