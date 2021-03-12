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

    nvinfer1::ITensor* input0{nullptr};
    nvinfer1::ITensor* input1{nullptr};

    // 两个输入都是正常的ITensor*
    if (input_tensors.size() == 2) {
      input0 = input_tensors[0];
      input1 = input_tensors[1];
    } else if (input_tensors.size() == 1) {
      // 输入中的一个是常量
      if (element_wise_desc->inputs[0].inUse) {
        input1 = input_tensors[0];

        const auto input0_layer =
            network->addConstant(TrtUtils::BroadcastDims(element_wise_desc->inputs[0].dim,
                                                         input1->getDimensions().nbDims),
                                 nvinfer1::Weights(element_wise_desc->inputs[0].data));
        if (input0_layer == nullptr) {
          LOG(ERROR) << "Create Network: Fail to create [element "
                        "wise::constant::0] layer";
          return {};
        }
        input0 = input0_layer->getOutput(0);
      } else {
        input0 = input_tensors[0];

        const auto input1_layer =
            network->addConstant(TrtUtils::BroadcastDims(element_wise_desc->inputs[1].dim,
                                                         input0->getDimensions().nbDims),
                                 nvinfer1::Weights(element_wise_desc->inputs[1].data));

        if (input1_layer == nullptr) {
          LOG(ERROR) << "Create Network: Fail to create [element "
                        "wise::constant::1] layer";
          return {};
        }
        input1 = input1_layer->getOutput(0);
      }
    } else {
      // 两个常量的情况
      auto broadcast_nbDims =
          element_wise_desc->inputs[0].dim.nbDims > element_wise_desc->inputs[1].dim.nbDims
              ? element_wise_desc->inputs[0].dim.nbDims
              : element_wise_desc->inputs[1].dim.nbDims;

      const auto input0_layer = network->addConstant(
          TrtUtils::BroadcastDims(element_wise_desc->inputs[0].dim, broadcast_nbDims),
          nvinfer1::Weights(element_wise_desc->inputs[0].data));

      const auto input1_layer = network->addConstant(
          TrtUtils::BroadcastDims(element_wise_desc->inputs[1].dim, broadcast_nbDims),
          nvinfer1::Weights(element_wise_desc->inputs[1].data));

      input0 = input0_layer->getOutput(0);
      input1 = input1_layer->getOutput(0);
    }

    // 同步两个向量的维度数量 (目前只用于fp_model)
    if (input0->getDimensions().nbDims > input1->getDimensions().nbDims) {
      auto reshape_input1 = network->addShuffle(*input1);
      reshape_input1->setReshapeDimensions(
          TrtUtils::BroadcastDims(input1->getDimensions(), input0->getDimensions().nbDims));
      input1 = reshape_input1->getOutput(0);
    }

    nvinfer1::IElementWiseLayer* element_wise =
        network->addElementWise(*input0, *input1, element_wise_desc->operation);
    if (!element_wise) {
      LOG(ERROR) << "Create Network: Fail to create [element wise] layer";
      return {};
    }
    return {element_wise->getOutput(0)};
  };
};

FWD_TRT_NAMESPACE_END
