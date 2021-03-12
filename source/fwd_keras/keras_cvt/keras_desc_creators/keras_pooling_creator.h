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
 * \brief Pooling 池化层描述创建器
 */
template <>
class TLayerDescCreator<TrtPoolingDesc> : public ILayerDescCreator {
 public:
  // TODO(Ao Li): 支持 Pooling ND
  bool Check(const Layer& layer) override {
    const std::string name = layer.Type();
    return name == "MaxPooling2D" || name == "AveragePooling2D";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtPoolingDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    input_names = layer.Inputs();

    const std::string class_name = layer.Type();

    // config
    const std::string data_format = layer.GetAttr<std::string>("data_format");
    const std::string dtype = layer.GetAttr<std::string>("dtype");
    const std::vector<int> pool_size = layer.GetAttr<std::vector<int>>("pool_size");
    const std::vector<int> strides = layer.GetAttr<std::vector<int>>("strides");
    const std::string padding = layer.GetAttr<std::string>("padding");

    T_CHECK_EQ(data_format, "channels_last");
    T_CHECK_EQ(dtype, "float32");

    auto layer_desc = std::make_shared<TrtPoolingDesc>();

    if (padding == "same") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } else if (padding == "valid") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    } else {
      LOG(ERROR) << "Unsupported padding mode " << padding;
      return nullptr;
    }

    layer_desc->windowSize = TrtUtils::ToDims(pool_size);
    layer_desc->stride = TrtUtils::ToDims(strides);

    if (class_name == "MaxPooling2D") {
      layer_desc->poolingType = nvinfer1::PoolingType::kMAX;
    } else if (class_name == "AveragePooling2D") {
      layer_desc->poolingType = nvinfer1::PoolingType::kAVERAGE;
    } else {
      LOG(ERROR) << "Unsupported pooling type " << class_name;
      return nullptr;
    }

    // zero-padded
    layer_desc->padding.nbDims = 0;
    layer_desc->averageCountExcludesPadding = true;

    return layer_desc;
  }
};

FWD_KERAS_NAMESPACE_END
