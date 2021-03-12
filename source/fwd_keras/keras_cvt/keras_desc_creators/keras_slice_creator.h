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
 * \brief Slice 描述创建器
 */
template <>
class TLayerDescCreator<TrtSliceDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override {
    std::string type = layer.Type();
    return type == "Cropping2D";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtSliceDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    input_names = layer.Inputs();

    const std::string layer_name = layer.Name();

    const std::string data_format = layer.GetAttr<std::string>("data_format");
    T_CHECK_EQ(data_format, "channels_last");

    const auto crops = layer.GetAttr<json>("cropping");

    const std::vector<int> h_crop = crops.at(0);
    const std::vector<int> w_crop = crops.at(1);

    auto layer_desc = std::make_shared<TrtSliceDesc>();

    layer_desc->stride = {4, {1, 1, 1, 1}};
    layer_desc->start = {4, {0, 0, h_crop[0], w_crop[0]}};  // NCHW
    layer_desc->size = {4,
                        {-1, -1, -h_crop[0] - h_crop[1] - 1, -w_crop[0] - w_crop[1] - 1}};  // NCHW

    return layer_desc;
  }
};

FWD_KERAS_NAMESPACE_END
