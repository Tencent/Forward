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
#include <unordered_map>
#include <vector>

#include "fwd_keras/keras_cvt/keras_desc_creators/i_keras_layer_creator.h"

FWD_KERAS_NAMESPACE_BEGIN
/**
 * \brief Activation 激活层描述创建器
 */
template <>
class TLayerDescCreator<TrtActivationDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override {
    std::string type = layer.Type();
    return type == "Activation";  // NK2AT_MAPPING.find(type) != NK2AT_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtActivationDesc::Create";

    input_names = layer.Inputs();
    const std::string type = layer.GetAttr<std::string>("activation");

    if (type == "softmax") {
      auto layer_desc = std::make_shared<TrtSoftmaxDesc>();

      // TODO(Ao Li): 目前只支持 NCHW softmax
      // tensorflow softmax 默认在 -1 维度，对应到 NCHW 是第一个维度
      layer_desc->axes = 1 << 1;
      return layer_desc;
    } else {
      auto layer_desc = std::make_shared<TrtActivationDesc>();
      auto ac_type = NK2AT_MAPPING.find(type);
      if (ac_type == NK2AT_MAPPING.end()) {
        LOG(ERROR) << "Unsupported activation type : " << type;
        return nullptr;
      }

      layer_desc->activationType = NK2AT_MAPPING.find(type)->second;
      return layer_desc;
    }
  }

 private:
  const std::unordered_map<std::string, nvinfer1::ActivationType> NK2AT_MAPPING = {
      {"relu", nvinfer1::ActivationType::kRELU},        // 0
      {"sigmoid", nvinfer1::ActivationType::kSIGMOID},  // 1
      {"tanh", nvinfer1::ActivationType::kTANH},        // 2
      {"elu", nvinfer1::ActivationType::kELU},          // 3
  };
};

FWD_KERAS_NAMESPACE_END
