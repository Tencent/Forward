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
 * \brief TRT 可分离卷积层创建器
 */
template <>
class TLayerCreator<TrtSeparableConvDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtSeparableConvDesc::CreateLayer";
    const auto convolution_desc = dynamic_cast<const TrtSeparableConvDesc*>(layer_desc);
    T_CHECK(convolution_desc);

    auto& input = *input_tensors[0];
    nvinfer1::IConvolutionLayer* convolution = network->addConvolutionNd(
        input, convolution_desc->nbDepthOutputMaps, convolution_desc->kernelSize,
        convolution_desc->depthKernelWeights, {nvinfer1::DataType::kFLOAT, nullptr, 0});

    if (convolution == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [convolution] layer.";
      return {};
    }

    convolution->setStrideNd(convolution_desc->stride);
    // convolution->setPaddingNd(convolution_desc->padding);
    convolution->setDilationNd(convolution_desc->dilation);

    if (convolution_desc->prePadding.nbDims) {
      convolution->setPrePadding(convolution_desc->prePadding);
    }
    if (convolution_desc->postPadding.nbDims) {
      convolution->setPostPadding(convolution_desc->postPadding);
    }

    convolution->setPaddingMode(convolution_desc->paddingMode);
    convolution->setNbGroups(convolution_desc->nbGroups);

    // point-wise convolution
    convolution = network->addConvolutionNd(
        *convolution->getOutput(0), convolution_desc->nbPointOutputMaps, {2, {1, 1}},  // point-wise
        convolution_desc->pointKernelWeights, convolution_desc->biasWeights);

    return {convolution->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
