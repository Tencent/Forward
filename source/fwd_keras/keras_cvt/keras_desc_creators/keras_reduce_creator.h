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
 * \brief Reduce 层描述创建器
 */
template <>
class TLayerDescCreator<TrtReduceDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override {
    const std::string name = layer.Type();
    if (name == "TensorFlowOpLayer") {
      const std::string op_name = layer.GetAttr<json>("node_def").at("op");
      return OT2RO_MAPPING.find(op_name) != OT2RO_MAPPING.end();
    }
    return false;
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtReduceDesc::Create";

    input_names = layer.Inputs();

    const std::string layer_name = layer.Name();

    const auto node_def = layer.GetAttr<json>("node_def");

    const std::string op_name = node_def.at("op");
    const bool keep_dims = node_def.at("attr").at("keep_dims").at("b");
    const auto dims = layer.GetAttr<json>("constants").at("1");

    auto layer_desc = std::make_shared<TrtReduceDesc>();

    layer_desc->keepDimensions = keep_dims;
    layer_desc->operation = OT2RO_MAPPING.find(op_name)->second;

    if (dims.is_number_integer()) {
      layer_desc->reduceAxes |= (1 << TrtUtils::NHWC2NCHWDim(dims.get<int>()));
    } else if (dims.is_array()) {
      const std::vector<int> real_dims = dims;
      for (auto dim : real_dims) {
        layer_desc->reduceAxes |= (1 << TrtUtils::NHWC2NCHWDim(dim));
      }
    }

    return layer_desc;
  }

 private:
  const std::unordered_map<std::string, nvinfer1::ReduceOperation> OT2RO_MAPPING = {
      {"Mean", nvinfer1::ReduceOperation::kAVG},
      {"Sum", nvinfer1::ReduceOperation::kSUM},
      {"Max", nvinfer1::ReduceOperation::kMAX},
      {"Min", nvinfer1::ReduceOperation::kMIN},
  };
};

FWD_KERAS_NAMESPACE_END
