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
 * \brief ElementWise 层描述创建器
 */
template <>
class TLayerDescCreator<TrtElementWiseDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override {
    return NK2EWP_MAPPING.find(layer.Type()) != NK2EWP_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtElementWiseDesc::Create";

    input_names = layer.Inputs();

    // TODO(Ao Li): 暂不支持常量算术运算
    if (input_names.size() != 2) {
      LOG(ERROR) << "Unsupported element-wise operation on constant tensor";
      return nullptr;
    }

    auto layer_desc = std::make_shared<TrtElementWiseDesc>();
    layer_desc->operation = NK2EWP_MAPPING.find(layer.Type())->second;

    return layer_desc;
  }

 private:
  const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> NK2EWP_MAPPING = {
      {"Add", nvinfer1::ElementWiseOperation::kSUM},        // 0
      {"Multiply", nvinfer1::ElementWiseOperation::kPROD},  // 1
      {"Subtract", nvinfer1::ElementWiseOperation::kSUB},   // 2
  };
};

FWD_KERAS_NAMESPACE_END
