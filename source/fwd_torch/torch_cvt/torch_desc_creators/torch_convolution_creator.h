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
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Convolution 层描述创建器
 * 这个类处理 Torch 的 c10::aten::_convolution 节点
 * 根据 tranposed 的情况生成 TrtConvolutionDesc 和 TrtDeconvolutionDesc
 * 两种层描述
 */
class TorchConvolutionCreator : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::_convolution;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TorchConvolutionCreator::Create";

    const auto inputs = node->inputs();
    T_CHECK(inputs.size() == 12 || inputs.size() == 13);

    const bool transposed = module.Get(node->inputs()[6]).toBool();
    if (!transposed) {
      auto layer_desc = TCreate<TrtConvolutionDesc>(node, module, input_values);

      const auto dilation = module.Get(inputs[5]).toIntList();
      layer_desc->dilation = ToDims(dilation);

      return layer_desc;
    } else {
      return TCreate<TrtDeconvolutionDesc>(node, module, input_values);
    }
  }

 private:
  /**
   * \brief 根据 描述类型 创建层描述
   * \tparam TLayerDesc 描述类型：Conv/Deconv
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  template <typename TLayerDesc>
  std::shared_ptr<TLayerDesc> TCreate(const JitNode* node, const TorchModule& module,
                                      std::vector<const JitValue*>& input_values) {
    const auto inputs = node->inputs();

    // Input 0, kind = input
    // Input 1, kind = c10::prim::GetAttr, weights
    // Input 2, kind = c10::prim::GetAttr, bias
    // Input 3, kind = c10::prim::ListConstruct, stride
    // Input 4, kind = c10::prim::ListConstruct, padding
    // Input 5, kind = c10::prim::ListConstruct, dilation
    // Input 6, kind = c10::prim::Constant, transposed
    // Input 7, kind = c10::prim::ListConstruct, output_padding
    // Input 8, kind = c10::prim::Constant, groups
    // Input 9, kind = c10::prim::Constant, benchmark
    // Input 10, kind = c10::prim::Constant, deterministic
    // Input 11, kind = c10::prim::Constant, cudnn_enabled
    // Input 12(torch1.7.0), kind = c10::prim::Constant, allow_tf32

    input_values.push_back(inputs[0]);

    auto layer_desc = std::make_shared<TLayerDesc>();

    const auto& weights_value = module.Get(inputs[1]);
    at::Tensor weights = weights_value.toTensor().contiguous();

    // dim=4 <=> (De)Conv_2d; dim=5 <=> (De)Conv_3d
    CHECK_GE(weights.ndimension(), 4);
    CHECK_LE(weights.ndimension(), 5);

    const bool transposed = module.Get(node->inputs()[6]).toBool();
    layer_desc->nbOutputMaps = weights.size(transposed ? 1 : 0);

    if (weights.ndimension() == 4) {
      layer_desc->kernelSize = {
          2, {static_cast<int>(weights.size(2)), static_cast<int>(weights.size(3))}};
    } else {
      layer_desc->kernelSize = {
          3,
          {static_cast<int>(weights.size(2)), static_cast<int>(weights.size(3)),
           static_cast<int>(weights.size(4))}};
    }
    layer_desc->kernelWeights = ToFwdWeights(weights);

    const auto& bias_value = module.Get(inputs[2]);
    if (!bias_value.isNone()) {
      const auto bias = bias_value.toTensor();
      layer_desc->biasWeights = ToFwdWeights(bias);
    }

    const auto stride = module.Get(inputs[3]).toIntList();
    layer_desc->stride = ToDims(stride);

    // 实际只有deconvolution时 output_padding 可能非0
    const auto padding = module.Get(inputs[4]).toIntList();
    const auto output_padding = module.Get(inputs[7]).toIntList();
    layer_desc->prePadding = ToDims(padding);
    layer_desc->postPadding = ToDims(padding) - ToDims(output_padding);

    layer_desc->nbGroups = module.Get(inputs[8]).toInt();
    if (transposed && layer_desc->nbOutputMaps % layer_desc->nbGroups != 0) {
      LOG(WARNING) << "group count(" << layer_desc->nbGroups
                   << ") do not divide output channel count(" << layer_desc->nbOutputMaps
                   << "), reset output channel count to " << layer_desc->nbGroups;
      layer_desc->nbOutputMaps = layer_desc->nbGroups;
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
