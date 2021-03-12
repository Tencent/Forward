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
 * \brief Separable Convolution 卷积层描述创建器
 */
template <>
class TLayerDescCreator<TrtSeparableConvDesc> : public ILayerDescCreator {
 public:
  // TODO(Ao Li): 支持 Convolution ND
  bool Check(const Layer& layer) override { return layer.Type() == "SeparableConv2D"; }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtSeparableConvDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    input_names = layer.Inputs();

    const std::string layer_name = layer.Name();

    // config
    const std::string data_format = layer.GetAttr<std::string>("data_format");
    const std::string dtype = layer.GetAttr<std::string>("dtype");
    const std::vector<int> kernel_size = layer.GetAttr<std::vector<int>>("kernel_size");
    const std::vector<int> strides = layer.GetAttr<std::vector<int>>("strides");
    const std::vector<int> dilation_rate = layer.GetAttr<std::vector<int>>("dilation_rate");
    const std::string padding = layer.GetAttr<std::string>("padding");
    const int filters = layer.GetAttr<int>("filters");
    const bool use_bias = layer.GetAttr<bool>("use_bias");

    T_CHECK_EQ(data_format, "channels_last");
    T_CHECK_EQ(dtype, "float32");

    auto layer_desc = std::make_shared<TrtSeparableConvDesc>();

    if (padding == "same") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } else if (padding == "valid") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    } else {
      LOG(ERROR) << "Unsupported padding mode " << padding;
      return nullptr;
    }

    KerasWeights depth_weights = reader.ReadWeight(layer_name, "depthwise_kernel:0");
    KerasWeights point_weights = reader.ReadWeight(layer_name, "pointwise_kernel:0");

    const auto depth_dims = depth_weights.GetDimension();

    layer_desc->nbDepthOutputMaps = depth_dims.d[2] * depth_dims.d[3];
    layer_desc->nbPointOutputMaps = filters;
    layer_desc->nbGroups = depth_dims.d[2];

    // NHWC 卷积核维度为 [kH, kW, in_c, out_c]
    // 而 TensorRT 卷积核参数维度为  [out_c, in_c, kH, kW], 因此这里需要
    // (3,2,0,1) 转置
    layer_desc->depthKernelWeights = depth_weights.ToFwdWeights();
    layer_desc->depthKernelWeights.Transpose(depth_dims, {3, 2, 0, 1});
    layer_desc->pointKernelWeights = point_weights.ToFwdWeights();
    layer_desc->pointKernelWeights.Transpose(point_weights.GetDimension(), {3, 2, 0, 1});

    layer_desc->kernelSize = TrtUtils::ToDims(kernel_size);
    layer_desc->dilation = TrtUtils::ToDims(dilation_rate);
    layer_desc->stride = TrtUtils::ToDims(strides);

    // TODO(Ao Li): 是否有存在 padding 的情况？
    layer_desc->prePadding.nbDims = 0;
    layer_desc->postPadding.nbDims = 0;

    if (use_bias) {
      KerasWeights bias = reader.ReadWeight(layer_name, "bias:0");
      layer_desc->biasWeights = bias.ToFwdWeights();
    }

    return layer_desc;
  }
};

FWD_KERAS_NAMESPACE_END
