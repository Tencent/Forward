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

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

// template <>
// class TLayerDescCreator<TrtNormalizationDesc> : public ILayerDescCreator {
//  public:
//   bool Check(const JitNode* node, const TorchModule& module) override;

//   std::shared_ptr<TrtLayerDesc> Create(
//       const JitNode* node, const TorchModule& module,
//       std::vector<const JitValue*>& input_values) override;

//  private:
//   std::vector<at::Tensor> tensor_storage;
// };

// TODO(Ao Li): 这部分代码需要重构

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Normalization 层描述创建器
 */
class TorchNormalizationCreator : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::instance_norm || node->kind() == c10::aten::batch_norm ||
           node->kind() == c10::aten::layer_norm;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    if (node->kind() == c10::aten::layer_norm) {
      return CreateLayerNorm(node, module, input_values);
    }

    // Batch Normalization and Instance Normalization

    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 9);
    // Input 0, kind = input
    // Input 1, kind = Tensor scales
    // Input 2, kind = Tensor bias
    // Input 3, kind = Tensor running_mean
    // Input 4, kind = Tensor running_var
    // Input 5, kind = bool use_input_stats(instance)/training(batch)
    // Input 6, kind = double momentum
    // Input 7, kind = double epsilon
    // Input 8, kind = bool cudnn_enabled

    const auto use_input_stats = module.Get(inputs[3]).isNone();
    if (use_input_stats) {
      return CreatePluginImpl(node, module, input_values);
    } else {
      return CreateScaleImpl(node, module, input_values);
    }
  }

 private:
  /**
   * \brief 创建 Layer Normalization 描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreateLayerNorm(const JitNode* node, const TorchModule& module,
                                                std::vector<const JitValue*>& input_values) const {
    LOG(INFO) << "TorchNormalizationCreator::CreateLayerNorm";
    const auto inputs = node->inputs();

    T_CHECK_EQ(inputs.size(), 6);

    // Input 0, kind = input
    // Input 1, kind = Int List normalized_shape
    // Input 2, kind = Tensor weight
    // Input 3, kind = Tensor bias
    // Input 4, kind = double epsilon
    // Input 5, kind = bool cudnn_enabled

    input_values.push_back(inputs[0]);

    const auto dummy = module.Get(inputs[0]).toTensor();
    const auto normalized_shape = ToIntVector(module.Get(inputs[1]));
    const auto eps = module.Get(inputs[4]).toDouble();

    auto layer_desc = std::make_shared<TrtNormalizationDesc>();

    layer_desc->use_input_stats = true;
    layer_desc->epsilon = eps;

    if (!module.Get(inputs[2]).isNone()) {
      T_CHECK(!module.Get(inputs[3]).isNone());
      const auto weights = module.Get(inputs[2]).toTensor().toType(c10::kFloat);
      const auto bias = module.Get(inputs[3]).toTensor().toType(c10::kFloat);
      layer_desc->affine = true;
      layer_desc->scales = Utils::ToFwdWeights(weights);
      layer_desc->bias = Utils::ToFwdWeights(bias);
    } else {
      layer_desc->affine = false;
    }

    if (dummy.ndimension() == 4) {  // NCHW
      T_CHECK_EQ(normalized_shape.size(), 3);
      T_CHECK_EQ(dummy.size(1), normalized_shape[0]);
      T_CHECK_EQ(dummy.size(2), normalized_shape[1]);
      T_CHECK_EQ(dummy.size(3), normalized_shape[2]);
      layer_desc->type = TrtNormalizationType::LAYER_NORMALIZATION;
    } else {
      T_CHECK(normalized_shape.size() == 1);
      layer_desc->type = TrtNormalizationType::SKIP_LAYER_NORMALIZATION;
      layer_desc->leading_dim = normalized_shape[0];
      // TODO(Ao Li): 注意这里没有 skip 直接使用 skip layer norm 的效率
      auto zeros = ::torch::zeros(dummy.sizes(), c10::ScalarType::Float);
      layer_desc->zeros = Utils::ToFwdWeights(zeros);
      layer_desc->use_fp16 =
          module.GetMode() == InferMode::HALF || module.GetMode() == InferMode::INT8;
      layer_desc->use_int8 = module.GetMode() == InferMode::INT8;
    }
    layer_desc->max_batch_size = module.GetMaxBatchSize();

    return layer_desc;
  }

  /**
   * \brief 创建 Plugin 版本的 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreatePluginImpl(const JitNode* node, const TorchModule& module,
                                                 std::vector<const JitValue*>& input_values) {
    LOG(INFO) << "TorchNormalizationCreator::CreatePluginImpl";

    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 9);

    auto layer_desc = std::make_shared<TrtNormalizationDesc>();

    input_values.push_back(inputs[0]);
    layer_desc->max_batch_size = module.GetMaxBatchSize();

    if (node->kind() == c10::aten::batch_norm) {
      layer_desc->type = TrtNormalizationType::BATCH_NORMALIZATION;
    } else if (node->kind() == c10::aten::instance_norm) {
      layer_desc->type = TrtNormalizationType::INSTANCE_NORMALIZATION;
    } else {
      LOG(ERROR) << "Unsupported normalziation type ";
      return nullptr;
    }

    layer_desc->affine = !(module.Get(inputs[1]).isNone());
    if (node->kind() == c10::aten::instance_norm) {
      layer_desc->use_input_stats = module.Get(inputs[5]).toBool();
    } else if (node->kind() == c10::aten::batch_norm) {
      layer_desc->use_input_stats = module.Get(inputs[3]).isNone();
    }
    layer_desc->epsilon = module.Get(inputs[7]).toDouble();

    if (layer_desc->affine) {
      const auto scales = module.Get(inputs[1]).toTensor();
      const auto bias = module.Get(inputs[2]).toTensor();

      layer_desc->scales = Utils::ToFwdWeights(scales);
      layer_desc->bias = Utils::ToFwdWeights(bias);
    }

    if (!layer_desc->use_input_stats) {
      const auto running_mean = module.Get(inputs[3]).toTensor();
      const auto running_var = module.Get(inputs[4]).toTensor();

      layer_desc->running_mean = Utils::ToFwdWeights(running_mean);
      layer_desc->running_var = Utils::ToFwdWeights(running_var);
    }

    return layer_desc;
  }

  /**
   * \brief 创建 TRT 原生支持的 Scale 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreateScaleImpl(const JitNode* node, const TorchModule& module,
                                                std::vector<const JitValue*>& input_values) {
    LOG(INFO) << "TorchNormalizationCreator::CreateScaleImpl";

    const auto inputs = node->inputs();
    T_CHECK_EQ(inputs.size(), 9);

    auto layer_desc = std::make_shared<TrtScaleDesc>();

    input_values.push_back(inputs[0]);

    const auto running_mean = module.Get(inputs[3]).toTensor();
    const auto running_var = module.Get(inputs[4]).toTensor();
    const auto eps = module.Get(inputs[7]).toDouble();

    auto scale_tensor = 1 / (running_var + eps).sqrt();
    auto shift_tensor = -running_mean * scale_tensor;

    // BatchNorm不会传入affine参数，所以统一自行判断
    if (!module.Get(inputs[1]).isNone()) {
      const auto weight = module.Get(inputs[1]).toTensor();
      const auto bias = module.Get(inputs[2]).toTensor();
      scale_tensor = scale_tensor * weight;
      shift_tensor = shift_tensor * weight + bias;
    }

    if (scale_tensor.scalar_type() != c10::ScalarType::Float ||
        shift_tensor.scalar_type() != c10::ScalarType::Float) {
      LOG(ERROR) << "scaler type or shift type not supported.";
      return nullptr;
    }

    layer_desc->shift = Utils::ToFwdWeights(shift_tensor);
    layer_desc->scale = Utils::ToFwdWeights(scale_tensor);
    layer_desc->mode = nvinfer1::ScaleMode::kCHANNEL;

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
