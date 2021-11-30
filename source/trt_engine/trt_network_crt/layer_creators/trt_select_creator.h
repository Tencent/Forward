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
    const auto select_desc = dynamic_cast<const TrtSelectDesc*>(layer_desc);
    T_CHECK(select_desc);

    auto& input = *input_tensors[0];
    auto nbDims = input.getDimensions().nbDims;

    // [Select]: condition tensor must have boolean type.
    nvinfer1::IIdentityLayer* identity = network->addIdentity(input);
    identity->getOutput(0)->setType(nvinfer1::DataType::kBOOL);
    nvinfer1::ITensor* condition_tensor = identity->getOutput(0);

    nvinfer1::ITensor* true_tensor;
    nvinfer1::ITensor* false_tensor;

    if (select_desc->inputs[1].inUse) {
      const auto true_const =
          network->addConstant(TrtUtils::BroadcastDims(select_desc->inputs[1].dim, nbDims),
                               nvinfer1::Weights(select_desc->inputs[1].data));
      true_tensor = true_const->getOutput(0);
    } else {
      true_tensor = input_tensors[1];
    }

    if (select_desc->inputs[2].inUse) {
      const auto false_const =
          network->addConstant(TrtUtils::BroadcastDims(select_desc->inputs[2].dim, nbDims),
                               nvinfer1::Weights(select_desc->inputs[2].data));
      false_tensor = false_const->getOutput(0);
    } else {
      false_tensor = input_tensors[2];
    }

    CHECK_EQ(true_tensor->getDimensions().nbDims, false_tensor->getDimensions().nbDims);

    const auto select = network->addSelect(*condition_tensor, *true_tensor, *false_tensor);
    select->setName((std::to_string(network->getNbLayers()) + std::string(" [Select]")).c_str());

    if (select == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [select] layer.";
      return {};
    }

    return {select->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
