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
 * \brief Shuffle 描述创建器
 */
template <>
class TLayerDescCreator<TrtShuffleDesc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override {
    const std::string name = layer.Type();
    return name == "Permute";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtShuffleDesc::Create";

    const std::string name = layer.Type();

    if (name == "Permute") {
      return CreatePermute(layer, input_names);
    }

    return nullptr;
  }

  std::shared_ptr<TrtLayerDesc> CreatePermute(const Layer& layer,
                                              std::vector<std::string>& input_names) const {
    input_names = layer.Inputs();
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    // config
    const std::string dtype = layer.GetAttr<std::string>("dtype");
    std::vector<int> dims = layer.GetAttr<std::vector<int>>("dims");

    T_CHECK_EQ(dtype, "float32");
    T_CHECK_EQ(dims.size(), 3);

    // Keras dims 参数中不包含样本维度
    dims.insert(dims.begin(), 0);

    auto layer_desc = std::make_shared<TrtShuffleDesc>();
    layer_desc->doFirstTrans = true;
    layer_desc->doReshape = false;
    layer_desc->doSecondTrans = false;
    layer_desc->firstTranspose =
        TrtUtils::ToPermutation(TrtUtils::NHWC2NCHWDim(TrtUtils::NHWC2NCHW(dims)));

    return layer_desc;
  }
};

FWD_KERAS_NAMESPACE_END
