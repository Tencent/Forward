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
 * \brief Reduce 层描述创建器
 */
template <>
class TLayerDescCreator<TrtReduceDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    if (node->kind() == c10::aten::mean || node->kind() == c10::aten::sum) {
      return true;
    }

    if ((node->kind() == c10::aten::max || node->kind() == c10::aten::min) &&
        (node->inputs().size() == 3 || node->inputs().size() == 1)) {
      return true;
    }

    if (node->kind() == c10::aten::var || node->kind() == c10::aten::std) {
      return true;
    }

    return false;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtReduceDesc::Create";

    if (node->kind() == c10::aten::var || node->kind() == c10::aten::std) {
      return CreateVarOrStd(node, module, input_values);
    }

    const auto inputs = node->inputs();

    input_values.push_back(inputs[0]);

    auto layer_desc = std::make_shared<TrtReduceDesc>();

    layer_desc->reduceAxes = 0;

    if (node->kind() == c10::aten::mean || node->kind() == c10::aten::sum) {
      // Input 0, kind = input
      // Input 1, kind = c10::prim::ListConstruct, dims / c10::prim::Constant?,
      // meaningless Input 2 (optional), kind = c10::prim::Construct, bool
      // keepDimensions Input 3 (optional), c10::prim::Constant?, meaningless
      layer_desc->operation = (node->kind() == c10::aten::mean ? nvinfer1::ReduceOperation::kAVG
                                                               : nvinfer1::ReduceOperation::kSUM);

      if (inputs.size() == 2) {
        layer_desc->keepDimensions = true;
      } else {
        const auto reduce_dim_list = module.Get(inputs[1]).toIntList();
        for (int i = 0; i < reduce_dim_list.size(); i++) {
          layer_desc->reduceAxes |= (1 << reduce_dim_list[i]);
        }

        layer_desc->keepDimensions = module.Get(inputs[2]).toBool();
      }
    } else if (node->kind() == c10::aten::max || node->kind() == c10::aten::min) {
      // Input 0, kind = input
      // Input 1, kind = c10::prim::Constant, dim
      // Input 2  kind = c10::prim::Constant, keepDimensions
      layer_desc->operation = (node->kind() == c10::aten::max ? nvinfer1::ReduceOperation::kMAX
                                                              : nvinfer1::ReduceOperation::kMIN);

      if (inputs.size() == 1) {
        layer_desc->keepDimensions = true;
      } else {
        const auto reduce_dim = module.Get(inputs[1]).toInt();
        layer_desc->reduceAxes = 1 << reduce_dim;
        layer_desc->keepDimensions = module.Get(inputs[2]).toBool();
      }
    }

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateVarOrStd(const JitNode* node, const TorchModule& module,
                                               std::vector<const JitValue*>& input_values) {
    LOG(INFO) << "TrtReduceDesc::CreateVar";

    const auto inputs = node->inputs();

    // torch.var(self, dim, unbiased, keepdim)
    input_values.push_back(inputs[0]);

    auto layer_desc = std::make_shared<TrtReduceDesc>();

    if (node->kind() == c10::aten::std) {
      layer_desc->isStdOp = true;
    } else {
      layer_desc->isVarOp = true;
    }
    layer_desc->reduceAxes = 0;

    const auto reduce_dims = module.Get(inputs[1]).toIntVector();
    for (auto dim : reduce_dims) {
      layer_desc->reduceAxes |= (1 << dim);
    }

    layer_desc->unbiased = module.Get(inputs[2]).toBool();
    layer_desc->keepDimensions = module.Get(inputs[3]).toBool();

    if (layer_desc->unbiased) {
      const auto dummy = module.Get(inputs[0]).toTensor();
      float n = 1.0f;
      for (auto dim : reduce_dims) {
        n *= dummy.size(dim);
      }
      T_CHECK_GT(n, 1);
      layer_desc->bias = n / (n - 1);
    }
    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
