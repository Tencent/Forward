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
#include <utility>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Shuffle 层描述创建器
 */
template <>
class TLayerDescCreator<TrtShuffleDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();

    return kind == c10::aten::flatten || kind == c10::aten::permute ||
           kind == c10::aten::pixel_shuffle || kind == c10::aten::transpose ||
           kind == c10::aten::view || kind == c10::aten::reshape || kind == c10::aten::squeeze ||
           kind == c10::aten::unsqueeze || kind == c10::aten::unsqueeze_ ||
           kind == c10::aten::expand;  // TODO(Ao Li): 暂时在这里处理 expand
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    const auto kind = node->kind();

    if (kind == c10::aten::flatten) return CreateFlatten(node, module, input_values);
    if (kind == c10::aten::permute) return CreatePermute(node, module, input_values);
    if (kind == c10::aten::transpose) return CreateTranspose(node, module, input_values);
    if (kind == c10::aten::expand) return CreateExpand(node, module, input_values);
    if (kind == c10::aten::pixel_shuffle) return CreatePixelShuffle(node, module, input_values);

    if (kind == c10::aten::view || kind == c10::aten::reshape)
      return CreateView(node, module, input_values);

    if (kind == c10::aten::unsqueeze || kind == c10::aten::unsqueeze_)
      return CreateUnSqueeze(node, module, input_values);

    if (kind == c10::aten::squeeze) return CreateSqueeze(node, module, input_values);

    return nullptr;
  }

 private:
  std::shared_ptr<TrtLayerDesc> CreatePixelShuffle(
      const JitNode* node, const TorchModule& module,
      std::vector<const JitValue*>& input_values) const {
    auto layer_desc = std::make_shared<TrtShuffleDesc>();

    // Input 0: Tensor input
    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 2);

    input_values.push_back(inputs[0]);

    const int upscale_factor = module.Get(inputs[1]).toInt();

    layer_desc->channel_block_size = upscale_factor;

    auto input_shape = torch_::ShapeOf(module.Get(inputs[0]).toTensor());
    const int nbdims = input_shape.size();
    input_shape[0] = 0;                                            // keep batch dim_idx
    input_shape[nbdims - 3] /= (upscale_factor * upscale_factor);  // C /= upscale_factor^2
    input_shape[nbdims - 2] *= upscale_factor;                     // H *= upscale_factor
    input_shape[nbdims - 1] *= upscale_factor;                     // W *= upscale_factor

    layer_desc->doReshape = true;
    layer_desc->reshapeDimensions = TrtUtils::ToDims(input_shape);

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateFlatten(const JitNode* node, const TorchModule& module,
                                              std::vector<const JitValue*>& input_values) const {
    auto layer_desc = std::make_shared<TrtShuffleDesc>();

    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 3);

    // Input 0: Tensor input
    // Input 1: int64_t start_dim
    // Input 2: int64_t end_dim

    input_values.push_back(inputs[0]);

    const int start_dim = module.Get(inputs[1]).toInt();
    const int end_dim = module.Get(inputs[2]).toInt();

    // 这个地方暂时还不好处理
    layer_desc->reshapeDimensions.nbDims = 2;
    layer_desc->reshapeDimensions.d[0] = start_dim;
    layer_desc->reshapeDimensions.d[1] = end_dim;

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreatePermute(const JitNode* node, const TorchModule& module,
                                              std::vector<const JitValue*>& input_values) const {
    LOG(INFO) << "TrtShuffleDescCreator::CreatePermute";

    auto layer_desc = std::make_shared<TrtShuffleDesc>();

    const auto inputs = node->inputs();
    // Input 0, kind = input
    // Input 1, kind = c10::prim::ListConstruct, permutes
    input_values.push_back(inputs[0]);
    const auto permutes = module.Get(inputs[1]).toIntList();

    layer_desc->doFirstTrans = true;
    layer_desc->doReshape = false;
    layer_desc->doSecondTrans = false;
    layer_desc->firstTranspose =
        TrtUtils::ToPermutation(std::vector<int>(permutes.begin(), permutes.end()));

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateTranspose(const JitNode* node, const TorchModule& module,
                                                std::vector<const JitValue*>& input_values) const {
    LOG(INFO) << "TrtShuffleDescCreator::CreateTranspose";

    auto layer_desc = std::make_shared<TrtShuffleDesc>();

    const auto inputs = node->inputs();
    // Input 0, kind = input
    // Input 1, kind = c10::prim::Constant, dim0
    // Input 2, kind = c10::prim::Constant, dim1
    input_values.push_back(inputs[0]);

    const at::Tensor dummy = module.Get(inputs[0]).toTensor();
    const auto dim0 = module.Get(inputs[1]).toInt();
    const auto dim1 = module.Get(inputs[2]).toInt();

    nvinfer1::Permutation permutes;
    for (int i = 0; i < dummy.ndimension(); ++i) {
      permutes.order[i] = i;
    }
    // auto permutes = TrtUtils::ToPermutation(DimsOf(dummy));
    std::swap(permutes.order[dim0], permutes.order[dim1]);

    layer_desc->doFirstTrans = true;
    layer_desc->doReshape = false;
    layer_desc->doSecondTrans = false;
    layer_desc->firstTranspose = permutes;

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateView(const JitNode* node, const TorchModule& module,
                                           std::vector<const JitValue*>& input_values) const {
    const auto inputs = node->inputs();

    // 处理常量 view
    if (inputs[0]->node()->kind() == c10::aten::zeros_like ||
        inputs[0]->node()->kind() == c10::prim::GetAttr ||
        inputs[0]->node()->kind() == c10::prim::Constant) {
      const at::Tensor input = module.Get(inputs[0]).toTensor().contiguous();
      auto constant_desc = std::make_shared<TrtConstantDesc>();
      constant_desc->weights = ToFwdWeights(input);
      constant_desc->dimensions = DimsOf(input);
      input_values.push_back(nullptr);  // constant input
      return constant_desc;
    }

    auto layer_desc = std::make_shared<TrtShuffleDesc>();

    // Input 0: Tensor input
    // Input 1: IntArrayRef size

    input_values.push_back(inputs[0]);

    const auto& value = module.Get(inputs[1]);

    layer_desc->doFirstTrans = false;
    layer_desc->doReshape = true;
    layer_desc->doSecondTrans = false;
    layer_desc->reshapeDimensions = ToDims(value.toIntList());

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateUnSqueeze(const JitNode* node, const TorchModule& module,
                                                std::vector<const JitValue*>& input_values) const {
    const auto inputs = node->inputs();

    // Input 0: Tensor input
    // Input 1: dim_idx
    auto dim = module.Get(inputs[1]).toInt();

    // 处理输入是常量的情况
    if (inputs[0]->node()->kind() == c10::prim::GetAttr ||
        inputs[0]->node()->kind() == c10::prim::Constant) {
      auto input = module.Get(inputs[0]).toTensor();
      input = input.unsqueeze(dim).contiguous();

      auto layer_desc = std::make_shared<TrtConstantDesc>();
      layer_desc->weights = torch_::ToFwdWeights(input);
      layer_desc->dimensions = torch_::DimsOf(input);

      input_values.push_back(nullptr);
      return layer_desc;
    }

    // 处理输入不是常量的情况
    input_values.push_back(inputs[0]);

    auto input_shape = torch_::ShapeOf(module.Get(inputs[0]).toTensor());
    if (dim < 0) {
      dim += input_shape.size() + 1;
    }
    CHECK_LE(dim, input_shape.size());

    input_shape.insert(input_shape.begin() + dim, 1);

    auto layer_desc = std::make_shared<TrtShuffleDesc>();
    layer_desc->doFirstTrans = false;
    layer_desc->doReshape = true;
    layer_desc->doSecondTrans = false;
    layer_desc->reshapeDimensions = TrtUtils::ToDims(input_shape);

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateSqueeze(const JitNode* node, const TorchModule& module,
                                              std::vector<const JitValue*>& input_values) const {
    const auto inputs = node->inputs();

    // Input 0: Tensor input
    // Input 1: dim_idx
    auto dim_idx = module.Get(inputs[1]).toInt();

    // 处理输入是常量的情况
    if (inputs[0]->node()->kind() == c10::prim::GetAttr ||
        inputs[0]->node()->kind() == c10::prim::Constant) {
      auto input = module.Get(inputs[0]).toTensor();
      input = input.squeeze(dim_idx).contiguous();

      auto layer_desc = std::make_shared<TrtConstantDesc>();
      layer_desc->weights = torch_::ToFwdWeights(input);
      layer_desc->dimensions = torch_::DimsOf(input);

      input_values.push_back(nullptr);
      return layer_desc;
    }

    // 处理输入不是常量的情况
    input_values.push_back(inputs[0]);

    auto input_shape = torch_::ShapeOf(module.Get(inputs[0]).toTensor());
    if (dim_idx < 0) {
      dim_idx += input_shape.size();
    }
    CHECK_LE(dim_idx, input_shape.size());
    CHECK_EQ(input_shape[dim_idx], 1);    // dim should be 1

    // get new shape
    std::vector<int64_t> new_shape;
    for (int i = 0; i < input_shape.size(); ++i) {
      if (i == dim_idx) continue;
      new_shape.push_back(input_shape[i]);
    }

    // permute dim_idx to last_dim
    std::vector<int> permute{0, 1, 2, 3, 4, 5, 6, 7};
    permute[dim_idx] = input_shape.size() - 1;
    permute[input_shape.size() - 1] = dim_idx;

    auto layer_desc = std::make_shared<TrtShuffleDesc>();
    layer_desc->doFirstTrans = true;
    layer_desc->firstTranspose = TrtUtils::ToPermutation(permute);
    layer_desc->doReshape = true;
    layer_desc->doSecondTrans = false;
    layer_desc->reshapeDimensions = TrtUtils::ToDims(new_shape);

    return layer_desc;
  }

  std::shared_ptr<TrtLayerDesc> CreateExpand(const JitNode* node, const TorchModule& module,
                                             std::vector<const JitValue*>& input_values) const {
    const auto inputs = node->inputs();

    // Input 0: Tensor input
    // Input 1: IntArrayRef size
    // Input 2: implicit

    // Tensor expand(const Tensor& self, IntArrayRef size, bool implicit)
    // 不应该存在 implicit expands
    T_CHECK(inputs.size() == 3 && !module.Get(inputs[2]).toBool());

    // TODO(Ao Li): 目前只支持输入是常量的情况
    if (inputs[0]->node()->kind() == c10::prim::GetAttr ||
        inputs[0]->node()->kind() == c10::prim::Constant) {
      auto input = module.Get(inputs[0]).toTensor();
      const auto size = module.Get(inputs[1]).toIntVector();
      input = input.expand(size).contiguous();

      auto layer_desc = std::make_shared<TrtConstantDesc>();
      layer_desc->weights = torch_::ToFwdWeights(input);
      layer_desc->dimensions = torch_::DimsOf(input);

      input_values.push_back(nullptr);
      return layer_desc;
    }
    LOG(ERROR) << "Unsupported non-constant tensor inputs of c10::aten::expand";
    return nullptr;
  }
};

FWD_TORCH_NAMESPACE_END
