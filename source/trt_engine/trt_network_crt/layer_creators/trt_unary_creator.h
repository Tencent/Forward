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
 * \brief TRT 一元操作层创建器
 */
template <>
class TLayerCreator<TrtUnaryDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtUnaryDesc::CreateLayer";
    const auto unary_desc = dynamic_cast<const TrtUnaryDesc*>(layer_desc);
    T_CHECK(unary_desc);

    nvinfer1::ITensor* input;
    if (unary_desc->input.inUse) {
      input = network->addConstant(unary_desc->input.dim, unary_desc->input.data)->getOutput(0);
    } else {
      input = input_tensors[0];
    }

    if (!unary_desc->is_combined_operation) {
      return CreateSingleUnaryLayer(network, unary_desc->operation, input);
    } else {
      return CreateMultiUnaryLayers(network, unary_desc, input);
    }
  }

 private:
  ITensorVector CreateSingleUnaryLayer(nvinfer1::INetworkDefinition* network,
                                       const nvinfer1::UnaryOperation operation,
                                       nvinfer1::ITensor* input) {
    nvinfer1::IUnaryLayer* unary = network->addUnary(*input, operation);
    unary->setName((std::to_string(network->getNbLayers()) + std::string(" [Unary]")).c_str());

    if (unary == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [unary] layer.";
      return {};
    }

    return {unary->getOutput(0)};
  }

  // To handle ops like tf.math.rsqrt, which consists two stacked operations: SQRT and RECIP.
  // For the above case: SQRT should be performed before RECIP, thus RECIP is considered as a
  // combined operation.
  //
  // In general case, if an op includes unary operations A, B, and C in order, A is
  // stored in unary_desc->operation, B and C are stored in unary_desc->combined_operations, and
  // should be traversed and added reversely, as the network is built from the output to the
  // input.
  ITensorVector CreateMultiUnaryLayers(nvinfer1::INetworkDefinition* network,
                                       const TrtUnaryDesc* unary_desc, nvinfer1::ITensor* input) {
    const auto combined_ops = unary_desc->combined_operations;
    nvinfer1::IUnaryLayer* unary = network->addUnary(*input, combined_ops.back());
    unary->setName((std::to_string(network->getNbLayers()) + std::string(" [Unary]")).c_str());

    for (auto it = combined_ops.crbegin() + 1; it != combined_ops.crend(); ++it) {
      unary = network->addUnary(*(unary->getOutput(0)), *it);
      unary->setName((std::to_string(network->getNbLayers()) + std::string(" [Unary]")).c_str());
    }

    return CreateSingleUnaryLayer(network, unary_desc->operation, unary->getOutput(0));
  }
};

FWD_TRT_NAMESPACE_END
