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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <algorithm>
#include <string>

#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief 构造 ElementWise 操作, 包括张量逐点 加、减、乘、除、 比大小 和 张量与常数值的计算 操作
 */
template <>
class TLayerCreator<TrtElementWiseDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtElementWiseDesc::CreateLayer";
    const auto element_wise_desc = dynamic_cast<const TrtElementWiseDesc*>(layer_desc);
    T_CHECK(element_wise_desc);

    nvinfer1::ITensor* inputs[2];

    int max_dims = 1;
    int input_id = 0;
    for (int i = 0; i < 2; ++i) {
      inputs[i] = element_wise_desc->inputs[i].inUse ? nullptr : input_tensors[input_id++];
      const int nbdims = element_wise_desc->inputs[i].inUse ? 1 : inputs[i]->getDimensions().nbDims;
      max_dims = std::max(max_dims, nbdims);
    }

    for (int i = 0; i < 2; ++i) {
      if (element_wise_desc->inputs[i].inUse) {
        auto& static_in = element_wise_desc->inputs[i];

        const auto c_layer =
            network->addConstant(TrtUtils::BroadcastDims(static_in.dim, max_dims), static_in.data);
        if (c_layer == nullptr) {
          LOG(ERROR) << "Create Network: Fail to create [element "
                        "wise::constant::0] layer";
          return {};
        }

        inputs[i] = c_layer->getOutput(0);
      }
    }

    CHECK_EQ(inputs[0]->getDimensions().nbDims, inputs[1]->getDimensions().nbDims);

    nvinfer1::IElementWiseLayer* element_wise =
        network->addElementWise(*inputs[0], *inputs[1], element_wise_desc->operation);
    if (!element_wise) {
      LOG(ERROR) << "Create Network: Fail to create [element wise] layer";
      return {};
    }

    element_wise->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [ElementWise]")).c_str());

    if (element_wise_desc->operation == nvinfer1::ElementWiseOperation::kGREATER ||
        element_wise_desc->operation == nvinfer1::ElementWiseOperation::kLESS) {
      return HandleBooleanOutput(network, element_wise);
    }

    return {element_wise->getOutput(0)};
  };

 private:
  ITensorVector HandleBooleanOutput(nvinfer1::INetworkDefinition* network,
                                    nvinfer1::IElementWiseLayer* element_wise) {
    // We need to handle kBOOL data type as Forward only allows kFloat computations.
    // For nvinfer1::IIdentityLayer, the only valid tranformations supported are: (kFLOAT -> kHALF),
    // (kFLOAT -> kINT8), (kHALF -> kFLOAT) and (kINT8 -> kFLOAT).
    element_wise->getOutput(0)->setType(nvinfer1::DataType::kINT8);

    nvinfer1::IIdentityLayer* identity = network->addIdentity(*(element_wise->getOutput(0)));
    if (!identity) {
      LOG(ERROR) << "Create Network: Fail to create [identity] layer";
      return {};
    }

    identity->setOutputType(0, element_wise->getInput(0)->getType());
    identity->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Identity]")).c_str());

    return {identity->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
