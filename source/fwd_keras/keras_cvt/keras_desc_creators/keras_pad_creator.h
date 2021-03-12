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
 * \brief Padding 描述创建器
 */
template <>
class TLayerDescCreator<TrtConstantPadDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override { return layer.Type() == "ZeroPadding2D"; }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtConstantPadDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    input_names = layer.Inputs();

    const std::string data_format = layer.GetAttr<std::string>("data_format");
    const std::string dtype = layer.GetAttr<std::string>("dtype");

    const auto paddings = layer.GetAttr<json>("padding");

    const std::vector<int> h_padding = paddings.at(0);
    const std::vector<int> w_padding = paddings.at(1);

    T_CHECK_EQ(data_format, "channels_last");
    T_CHECK_EQ(dtype, "float32");
    T_CHECK_EQ(paddings.size(), 2);

    // [H0, H1], [W0, W1]
    T_CHECK_EQ(h_padding.size(), 2);
    T_CHECK_EQ(w_padding.size(), 2);

    auto layer_desc = std::make_shared<TrtConstantPadDesc>();

    layer_desc->value = 0.0f;
    layer_desc->dims = {w_padding[0], w_padding[1], h_padding[0], h_padding[1]};

    return layer_desc;
  }
};

FWD_KERAS_NAMESPACE_END
