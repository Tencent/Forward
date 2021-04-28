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

#include <vector>

#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT 层选择层创建器，目前只实质处理了bool转int型的情况
 */
template <>
class TLayerCreator<TrtSelectDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtSelectDesc::CreateLayer";
#if NV_TENSORRT_MAJOR >= 7
    const auto select_desc = dynamic_cast<const TrtSelectDesc*>(layer_desc);
    T_CHECK(select_desc);

    auto& input = *input_tensors[0];
    auto nbDims = input.getDimensions().nbDims;

    nvinfer1::ITensor* true_tensor;
    nvinfer1::ITensor* false_tensor;
    if (select_desc->trueInput.inUse) {
      const auto true_const =
          network->addConstant(TrtUtils::BroadcastDims(select_desc->trueInput.dim, nbDims),
                               nvinfer1::Weights(select_desc->trueInput.data));
      true_tensor = true_const->getOutput(0);
    } else {
      true_tensor = input_tensors[1];
    }

    if (select_desc->falseInput.inUse) {
      const auto false_const =
          network->addConstant(TrtUtils::BroadcastDims(select_desc->falseInput.dim, nbDims),
                               nvinfer1::Weights(select_desc->falseInput.data));
      false_tensor = false_const->getOutput(0);
    } else {
      false_tensor = input_tensors.back();
    }

    const auto select = network->addSelect(input, *true_tensor, *false_tensor);

    if (select == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [select] layer.";
      return {};
    }

    return {select->getOutput(0)};
#else // NV_TENSORRT_MAJOR < 7
    LOG(ERROR) << "Select Layer requires TENSORRT_VERSION >= 7.";
    return {};
#endif // NV_TENSORRT_MAJOR >= 7
  }
};

FWD_TRT_NAMESPACE_END
