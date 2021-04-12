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

#include <string>
#include <vector>

#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief 构造ElementWise操作, 包括张量逐点加减乘除和张量与常数值的计算操作
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

    // 同步两个向量的维度数量 (目前只用于fp_model)
    if (inputs[0]->getDimensions().nbDims > inputs[1]->getDimensions().nbDims) {
      auto reshape_input1 = network->addShuffle(*inputs[1]);
      reshape_input1->setReshapeDimensions(
          TrtUtils::BroadcastDims(inputs[1]->getDimensions(), inputs[0]->getDimensions().nbDims));
      inputs[1] = reshape_input1->getOutput(0);
    }

    nvinfer1::IElementWiseLayer* element_wise =
        network->addElementWise(*inputs[0], *inputs[1], element_wise_desc->operation);
    if (!element_wise) {
      LOG(ERROR) << "Create Network: Fail to create [element wise] layer";
      return {};
    }
    return {element_wise->getOutput(0)};
  };
};

FWD_TRT_NAMESPACE_END
