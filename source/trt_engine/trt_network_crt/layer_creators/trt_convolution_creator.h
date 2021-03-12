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
 * \brief TRT 卷积层创建器
 */
template <>
class TLayerCreator<TrtConvolutionDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtConvolutionNdDesc::CreateLayer";
    const auto convolution_desc = dynamic_cast<const TrtConvolutionDesc*>(layer_desc);
    T_CHECK(convolution_desc);

    nvinfer1::ITensor* input = input_tensors[0];

    // 1. tf 一维卷积的处理，需要先进行维度转换
    if (convolution_desc->squeeze_dim != -1) {
      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*input);
      shuffle->setFirstTranspose({0, 3, 1, 2});  // NHWC->NCHW
      input = shuffle->getOutput(0);
    }

    nvinfer1::IConvolutionLayer* convolution = network->addConvolutionNd(
        *input, convolution_desc->nbOutputMaps, convolution_desc->kernelSize,
        convolution_desc->kernelWeights, convolution_desc->biasWeights);

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

    nvinfer1::ITensor* output = convolution->getOutput(0);

    // 2. tf 一维卷积的处理，需要进行维度转换，需要将输出 Squeeze
    if (convolution_desc->squeeze_dim != -1) {
      nvinfer1::Dims dim = output->getDimensions();
      T_CHECK_EQ(dim.d[convolution_desc->squeeze_dim], 1);
      dim.d[2] = dim.d[1];
      dim.d[1] = dim.d[3];
      --dim.nbDims;
      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*output);
      shuffle->setFirstTranspose({0, 2, 3, 1});  // NCHW->NHWC
      shuffle->setReshapeDimensions(dim);
      output = shuffle->getOutput(0);
    }

    return {output};
  }
};

FWD_TRT_NAMESPACE_END
