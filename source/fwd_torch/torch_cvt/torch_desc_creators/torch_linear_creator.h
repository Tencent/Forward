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
 * \brief FullyConnected 和 Addmm 层描述创建器
 */
class TorchAddmmCreator : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    // LOG(INFO) << "TorchAddmmCreator::Check";

    return node->kind() == c10::aten::linear ||
           (node->kind() == c10::aten::addmm && node->inputs().size() == 5);
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TorchAddmmCreator::Create";

    const auto inputs = node->inputs();

    if (node->kind() == c10::aten::linear) {
      return CreateLinear(node, module, input_values);
    }

    if (inputs[0]->node()->kind() == c10::prim::GetAttr) {
      return CreateFullyConnected(node, module, input_values);
    }

    if (inputs[0]->node()->kind() == c10::prim::Param) {
      return CreateMatMulAdd(node, module, input_values);
    }

    LOG(ERROR) << "Unhandled c10::aten::addmm op";

    return nullptr;
  }

 private:
  /**
   * \brief 创建 Linear 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreateLinear(const JitNode* node, const TorchModule& module,
                                             std::vector<const JitValue*>& input_values) {
    const auto inputs = node->inputs();
    // %res = c10::aten::linear(%input, %weight, %bias)

    // 将输入返回
    input_values.push_back(inputs[0]);

    const auto weights = module.Get(inputs[1]).toTensor();
    const auto bias_value = module.Get(inputs[2]);

    T_CHECK(weights.ndimension() == 2);

    auto layer_desc = std::make_shared<TrtFullyConnectedDesc>();

    if (!bias_value.isNone()) {
      const auto bias = bias_value.toTensor();
      layer_desc->biasWeights = Utils::ToFwdWeights(bias);
      T_CHECK(bias.ndimension() == 1);
      T_CHECK_EQ(bias.numel(), weights.size(0));
    }

    layer_desc->nbOutputChannels = weights.size(0);
    layer_desc->kernelWeights = Utils::ToFwdWeights(weights);
    return layer_desc;
  }

  /**
   * \brief 创建 FullyConnected 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreateFullyConnected(const JitNode* node, const TorchModule& module,
                                                     std::vector<const JitValue*>& input_values) {
    const auto inputs = node->inputs();

    // Input node kind = c10::prim::GetAttr
    // Input node kind = input
    // Input node kind = c10::aten::t
    // Input node kind = c10::prim::Constant
    // Input node kind = c10::prim::Constant

    // 将输入返回
    input_values.push_back(inputs[1]);

    const auto& bias_value = module.Get(inputs[0]);
    at::Tensor bias = bias_value.toTensor();

    const auto& weights_value = module.Get(inputs[2]);
    // 这个地方要转置
    at::Tensor weights = weights_value.toTensor().t();

    T_CHECK(bias.ndimension() == 1);
    T_CHECK(weights.ndimension() == 2);
    T_CHECK_EQ(bias.numel(), weights.size(0));

    int beta = module.Get(inputs[3]).toInt();
    if (beta != 1) {
      bias *= beta;
    }

    int alpha = module.Get(inputs[4]).toInt();
    if (alpha != 1) {
      weights *= alpha;
    }

    auto layer_desc = std::make_shared<TrtFullyConnectedDesc>();

    layer_desc->nbOutputChannels = bias.numel();

    layer_desc->kernelWeights = Utils::ToFwdWeights(weights);
    layer_desc->biasWeights = Utils::ToFwdWeights(bias);

    return layer_desc;
  }

  /**
   * \brief 创建 MatMulAdd 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreateMatMulAdd(const JitNode* node, const TorchModule& module,
                                                std::vector<const JitValue*>& input_values) {
    const auto inputs = node->inputs();

    // Input node kind = c10::prim::Param bias
    // Input node kind = c10::prim::Param input
    // Input node kind = c10::prim::Param weights
    // Input node kind = c10::prim::Constant
    // Input node kind = c10::prim::Constant

    // 将 3 个输入按照input, weights, bias顺序返回
    input_values.push_back(inputs[1]);
    input_values.push_back(inputs[2]);
    input_values.push_back(inputs[0]);

    const int beta = module.Get(inputs[3]).toInt();
    const int alpha = module.Get(inputs[4]).toInt();

    auto layer_desc = std::make_shared<TrtMatMulAddDesc>();
    layer_desc->beta = beta;
    layer_desc->alpha = alpha;

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
