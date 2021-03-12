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

#include "fwd_keras/keras_cvt/keras_desc_creators/i_keras_layer_creator.h"

FWD_KERAS_NAMESPACE_BEGIN
/**
 * \brief BatchNorm 描述创建器
 */
template <>
class TLayerDescCreator<TrtNormalizationDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override { return layer.Type() == "BatchNormalization"; }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtNormalizationDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    input_names = layer.Inputs();

    const std::string layer_name = layer.Name();

    // config
    const std::string config_name = layer.GetAttr<std::string>("name");
    const std::string dtype = layer.GetAttr<std::string>("dtype");
    const int axis = layer.GetAttr<std::vector<int>>("axis")[0];
    const float epsilon = layer.GetAttr<float>("epsilon");
    const bool center = layer.GetAttr<bool>("center");
    const bool scale = layer.GetAttr<bool>("scale");

    if (axis != 3) {
      LOG(ERROR) << "Unsupported axis = " << axis << " in batch normalization";
      return nullptr;
    }
    T_CHECK_EQ(dtype, "float32");

    const auto mean_data = reader.ReadWeight(layer_name, "moving_mean:0").GetData();
    const auto var_data = reader.ReadWeight(layer_name, "moving_variance:0").GetData();

    T_CHECK_EQ(mean_data.size(), var_data.size());

    // 计算 scale & shift
    std::vector<float> scale_data;
    std::vector<float> shift_data;
    scale_data.reserve(mean_data.size());
    shift_data.reserve(mean_data.size());
    for (size_t i = 0; i < mean_data.size(); ++i) {
      scale_data.push_back(1.0f / sqrtf(var_data[i] + epsilon));
      shift_data.push_back(-mean_data[i] * scale_data[i]);
    }

    if (scale) {
      const auto gamma_data = reader.ReadWeight(layer_name, "gamma:0").GetData();
      T_CHECK_EQ(mean_data.size(), gamma_data.size());
      for (size_t i = 0; i < gamma_data.size(); ++i) {
        scale_data[i] *= gamma_data[i];
        shift_data[i] *= gamma_data[i];
      }
    }
    if (center) {
      const auto beta_data = reader.ReadWeight(layer_name, "beta:0").GetData();
      T_CHECK_EQ(mean_data.size(), beta_data.size());
      for (size_t i = 0; i < beta_data.size(); ++i) {
        shift_data[i] += beta_data[i];
      }
    }

    // keras batch norm use_input_stats 总是 false, 使用 scale 来实现
    auto layer_desc = std::make_shared<TrtScaleDesc>();

    layer_desc->mode = nvinfer1::ScaleMode::kCHANNEL;
    layer_desc->scale = FwdWeights(scale_data);
    layer_desc->shift = FwdWeights(shift_data);

    return layer_desc;
  }
};

FWD_KERAS_NAMESPACE_END
