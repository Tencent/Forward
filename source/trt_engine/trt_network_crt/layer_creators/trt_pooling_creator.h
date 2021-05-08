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
 * \brief TRT 池化层创建器
 */
template <>
class TLayerCreator<TrtPoolingDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtPoolingDesc::CreateLayer";
    const auto pooling_desc = dynamic_cast<const TrtPoolingDesc*>(layer_desc);
    T_CHECK(pooling_desc);

    auto& input = *input_tensors[0];
    nvinfer1::IPoolingLayer* pooling =
        network->addPoolingNd(input, pooling_desc->poolingType, pooling_desc->windowSize);

    if (pooling == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [pooling] layer.";
      return {};
    }

    pooling->setPaddingMode(pooling_desc->paddingMode);
    // nbDims = 0 when no stride
    if (pooling_desc->stride.nbDims) {
      pooling->setStrideNd(pooling_desc->stride);
    }
    // nbDims = 0 when no pad applied
    if (pooling_desc->padding.nbDims) {
      pooling->setPaddingNd(pooling_desc->padding);
    }
    if (pooling_desc->poolingType == nvinfer1::PoolingType::kAVERAGE) {
      pooling->setAverageCountExcludesPadding(pooling_desc->averageCountExcludesPadding);
    }
    return {pooling->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
