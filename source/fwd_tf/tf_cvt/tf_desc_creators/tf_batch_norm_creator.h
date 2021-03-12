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

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief BatchNorm 层描述创建器
 */
template <>
class TLayerDescCreator<TrtNormalizationDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();
    return type == "FusedBatchNormV3" || type == "FusedBatchNorm";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtNormalizationDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    T_CHECK_EQ(op.NumInputs(), 5);

    // Input 0, kind = input
    const auto input = op.Input(0);
    const auto scales = op.Input(1);
    const auto offset = op.Input(2);
    const auto mean = op.Input(3);
    const auto variance = op.Input(4);

    // 将输入返回
    op_inputs.push_back(input);

    // 属性
    const auto epsilon = op.GetAttrFloat("epsilon");
    const auto data_format = op.GetAttrString("data_format");
    T_CHECK_EQ(data_format, "NHWC");

    const auto mean_tensor = mean.GetConstantTensor();
    const auto var_tensor = variance.GetConstantTensor();
    const auto weights_tensor = scales.GetConstantTensor();
    const auto bias_tensor = offset.GetConstantTensor();

    // scale 实现： 在提供 running_mean & running_var 的情况下
    if (mean_tensor.ElementCount() > 0) {
      LOG(INFO) << "BatchNorm Use Scale Implementation";

      auto layer_desc = std::make_shared<TrtScaleDesc>();

      T_CHECK_EQ(mean_tensor.Type(), TF_FLOAT);
      T_CHECK_EQ(var_tensor.Type(), TF_FLOAT);
      T_CHECK_EQ(weights_tensor.Type(), TF_FLOAT);
      T_CHECK_EQ(bias_tensor.Type(), TF_FLOAT);

      const auto mean_size = mean_tensor.Size();
      const auto var_size = var_tensor.Size();
      const auto weights_size = weights_tensor.Size();
      const auto bias_size = bias_tensor.Size();

      T_CHECK_EQ(mean_size, var_size);

      const auto mean_data = mean_tensor.Data<float>();
      const auto var_data = var_tensor.Data<float>();
      const auto weights_data = weights_tensor.Data<float>();
      const auto bias_data = bias_tensor.Data<float>();

      // 计算 scale & shift
      std::vector<float> scale_data;
      std::vector<float> shift_data;
      scale_data.reserve(mean_size);
      shift_data.reserve(mean_size);
      for (size_t i = 0; i < mean_size; ++i) {
        scale_data.push_back(1.0f / sqrtf(var_data[i] + epsilon));
        shift_data.push_back(-mean_data[i] * scale_data[i]);
      }

      if (weights_size > 0) {
        T_CHECK_EQ(mean_size, weights_size);
        T_CHECK_EQ(mean_size, bias_size);

        for (size_t i = 0; i < weights_size; ++i) {
          scale_data[i] *= weights_data[i];
          shift_data[i] = shift_data[i] * weights_data[i] + bias_data[i];
        }
      }

      layer_desc->mode = nvinfer1::ScaleMode::kCHANNEL;
      layer_desc->scale = FwdWeights(scale_data);
      layer_desc->shift = FwdWeights(shift_data);

      return layer_desc;
    }

    // plugin 实现： 在不提供 running_mean & running_var 的情况下
    LOG(INFO) << "BatchNorm Use Plugin Implementation";

    auto layer_desc = std::make_shared<TrtNormalizationDesc>();

    layer_desc->use_input_stats = true;
    layer_desc->type = TrtNormalizationType::BATCH_NORMALIZATION;
    layer_desc->scales = Utils::ToFwdWeights(weights_tensor);
    layer_desc->bias = Utils::ToFwdWeights(bias_tensor);
    layer_desc->running_mean = Utils::ToFwdWeights(mean_tensor);
    layer_desc->running_var = Utils::ToFwdWeights(var_tensor);
    layer_desc->affine = layer_desc->scales.Count() > 0;
    layer_desc->epsilon = epsilon;

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
