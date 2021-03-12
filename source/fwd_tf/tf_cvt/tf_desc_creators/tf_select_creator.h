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
 * \brief Select 选择层描述创建器
 */
template <>
class TLayerDescCreator<TrtSelectDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    std::string type = op.OpType();

    if (type != "Cast" && type != "Mul") return false;

    const TF_DataType dtype = op.InputType(0);
    const TF_DataType o_dtype = op.OutputType(0);

    if (type == "Cast" && dtype == TF_DataType::TF_BOOL && o_dtype != TF_BOOL) {
      return true;
    }

    if (type == "Mul") {
      for (int i = 0; i < 2; i++) {
        auto input = op.Input(i);
        std::string input_type = input.OpType();
        TF_DataType input_dtype = input.InputType(0);
        TF_DataType output_dtype = input.OutputType(0);
        if (input_type == "Cast" && input_dtype == TF_DataType::TF_BOOL &&
            output_dtype != TF_BOOL) {
          return true;
        }
      }
    }
    return false;
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtSelectDesc::Create";

    std::string type = op.OpType();

    const TF_DataType dtype = op.InputType(0);

    const auto input = op.Input(0);

    if (type == "Cast" && dtype == TF_BOOL) {
      return CreateOneZeroSelect(op, graph, op_inputs);
    }

    auto layer_desc = std::make_shared<TrtSelectDesc>();

    // 先判断bool矩阵在哪边, 先传入bool, 然后thenInput, 最后elseInput(0)
    std::string input0_type = input.OpType();
    TF_DataType input_dtype = input.InputType(0);
    TF_DataType output_dtype = input.OutputType(0);
    if (input0_type == "Cast" && input_dtype == TF_DataType::TF_BOOL && output_dtype != TF_BOOL) {
      const auto cast_input = input.Input(0);
      op_inputs.push_back(cast_input);
      const auto input_true = op.Input(1);
      op_inputs.push_back(input_true);
    } else {
      const auto input_bool = op.Input(1);
      const auto cast_input = input_bool.Input(0);
      op_inputs.push_back(cast_input);
      op_inputs.push_back(input);
    }
    layer_desc->falseInput.inUse = true;
    layer_desc->falseInput.dim = {1, 1};

    auto output_type = op.OutputType(0);
    if (output_type == TF_INT32) {
      std::vector<int> zero(1, 0);
      layer_desc->falseInput.data = FwdWeights(zero);
    } else {
      std::vector<float> zero(1, 0);
      layer_desc->falseInput.data = FwdWeights(zero);
    }

    return layer_desc;
  }

 private:
  std::shared_ptr<TrtLayerDesc> CreateOneZeroSelect(const Operation& op, const Graph& graph,
                                                    std::vector<Output>& op_inputs) {
    std::string type = op.OpType();

    const auto input = op.Input(0);
    auto output_type = op.OutputType(0);

    auto layer_desc = std::make_shared<TrtSelectDesc>();
    layer_desc->trueInput.inUse = true;
    layer_desc->falseInput.inUse = true;
    layer_desc->trueInput.dim = {1, 1};
    layer_desc->falseInput.dim = {1, 1};
    if (output_type == TF_INT32) {
      std::vector<int> one(1, 1);
      std::vector<int> zero(1, 0);
      layer_desc->trueInput.data = FwdWeights(one);
      layer_desc->falseInput.data = FwdWeights(zero);
    } else {
      std::vector<float> one(1, 1);
      std::vector<float> zero(1, 0);
      layer_desc->trueInput.data = FwdWeights(one);
      layer_desc->falseInput.data = FwdWeights(zero);
    }
    op_inputs.push_back(input);

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
