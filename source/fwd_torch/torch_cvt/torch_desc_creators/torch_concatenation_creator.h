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
#include <utility>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Concatenation 层描述创建器
 */
template <>
class TLayerDescCreator<TrtConcatenationDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::cat || node->kind() == c10::aten::stack;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    // 针对 split stack 模式进行优化
    if (CheckSplitStack(node)) {
      return CreateSplitStack(node, module, input_values);
    }

    return CreateConcatenation(node, module, input_values);
  }

 private:
  std::shared_ptr<TrtLayerDesc> CreateConcatenation(
      const JitNode* node, const TorchModule& module,
      std::vector<const JitValue*>& input_values) const {
    LOG(INFO) << "TrtConcatenationDescCreator::CreateConcatenation";

    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 2);

    // Input 0: TensorList tensors
    // Input 1: int64_t dim
    {
      const auto tensors_node = inputs[0]->node();
      T_CHECK(tensors_node->kind() == c10::prim::ListConstruct);

      for (auto input : tensors_node->inputs()) {
        ExtractAllInputs(input, input_values);
      }
    }

    auto layer_desc = std::make_shared<TrtConcatenationDesc>();
    layer_desc->axis = module.Get(inputs[1]).toInt();
    layer_desc->is_stack = node->kind() == c10::aten::stack;

    return layer_desc;
  }

  bool CheckSplitStack(const JitNode* node) {
    if (node->kind() == c10::aten::stack) {
      const auto list_construct = node->inputs()[0]->node();
      if (list_construct->kind() == c10::prim::ListConstruct) {
        const auto list_unpack = list_construct->inputs()[0]->node();
        if (list_unpack->kind() == c10::prim::ListUnpack) {
          const auto split = list_unpack->input()->node();
          if (split->kind() == c10::aten::split) {
            split_node_ = split;
            return true;
          }
        }
      }
    }
    return false;
  }

  std::shared_ptr<TrtLayerDesc> CreateSplitStack(const JitNode* node, const TorchModule& module,
                                                 std::vector<const JitValue*>& input_values) const {
    LOG(INFO) << "TrtConcatenationDescCreator::CreateSplitStack";

    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 2);
    T_CHECK_NOTNULL(split_node_);

    input_values.push_back(split_node_->inputs()[0]);

    const auto split_size = module.Get(split_node_->inputs()[1]).toInt();
    const auto split_dim = module.Get(split_node_->inputs()[2]).toInt();
    const auto stack_dim = module.Get(inputs[1]).toInt();

    // reshape
    const auto& dummy = module.Get(split_node_->inputs()[0]).toTensor();
    const nvinfer1::Dims input_dim = DimsOf(dummy);
    const auto sections = input_dim.d[split_dim] / split_size;
    nvinfer1::Dims reshape_dim{input_dim.nbDims + 1,
                               {
                                   0,
                               }};
    for (int i = 0, k = 0; i < input_dim.nbDims; ++i) {
      if (i == split_dim) {
        reshape_dim.d[k++] = sections;
        reshape_dim.d[k++] = split_size;
      } else {
        reshape_dim.d[k++] = input_dim.d[i];
      }
    }

    // permutation
    nvinfer1::Permutation perm{};
    for (int i = 0; i < reshape_dim.nbDims; ++i) {
      perm.order[i] = i;
    }
    std::swap(perm.order[stack_dim], perm.order[split_dim]);

    auto layer_desc = std::make_shared<TrtShuffleDesc>();
    layer_desc->doFirstTrans = false;
    layer_desc->doReshape = true;
    layer_desc->doSecondTrans = true;
    layer_desc->reshapeDimensions = reshape_dim;
    layer_desc->secondTranspose = perm;
    return layer_desc;
  }

  const JitNode* split_node_{nullptr};
};

FWD_TORCH_NAMESPACE_END
