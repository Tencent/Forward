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

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief MatMul 层描述创建器
 */
template <>
class TLayerDescCreator<TrtMatrixMultiplyDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::bmm || node->kind() == c10::aten::matmul;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& node_inputs) override {
    const auto inputs = node->inputs();

    // Input 0: Tensor input0
    // Input 1: Tensor input1
    auto layer_desc = std::make_shared<TrtMatrixMultiplyDesc>();

    for (int i = 0; i < 2; i++) {
      const auto input_dims = module.Get(inputs[i]).toTensor().ndimension();
      if (inputs[i]->node()->kind() == c10::prim::GetAttr) {
        auto ivalue = module.Get(inputs[i]);
        T_CHECK(ivalue.isTensor());
        layer_desc->inputs[i].inUse = true;
        layer_desc->inputs[i].data = ToFwdWeights(ivalue.toTensor());
        layer_desc->inputs[i].dim = layer_desc->inputs[i].data.Dims();
      } else if (CheckTransposed(module, input_dims, inputs[i]->node())) {
        layer_desc->op[i] = nvinfer1::MatrixOperation::kTRANSPOSE;
        node_inputs.push_back(inputs[i]->node()->inputs()[0]);
      } else {
        node_inputs.push_back(inputs[i]);
      }
    }

    return layer_desc;
  }

  /**
   * \brief 检查该输入是否需要转置
   * \param module 常量获取器
   * \param input_dims 输入维度
   * \param child 子节点（输入）
   * \return
   */
  bool CheckTransposed(const TorchModule& module, const int64_t input_dims, const JitNode* child) {
    if (child->kind() == c10::aten::permute || child->kind() == c10::aten::transpose) {
      // redundancy check
      if (child->kind() == c10::aten::permute) {
        auto shape = module.Get(child->inputs()[1]).toIntList();
        if (shape.size() == 3 && shape[0] == 0 && shape[1] == 2 && shape[2] == 1) {
          return true;
        }
      } else if (child->kind() == c10::aten::transpose) {
        const auto dim0 = module.Get(child->inputs()[1]).toInt();
        const auto dim1 = module.Get(child->inputs()[2]).toInt();
        if (input_dims == 3 && (dim0 == 1 && dim1 == 2 || dim0 == 2 && dim1 == 1) ||
            input_dims == 4 && (dim0 == -1 && dim1 == -2 || dim0 == -2 && dim1 == -1)) {
          return true;
        }
      }
    }
    return false;
  }
};

FWD_TORCH_NAMESPACE_END
