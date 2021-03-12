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
 * \brief TRT 反卷积层创建器
 */
template <>
class TLayerCreator<TrtDeconvolutionDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtDeconvolutionNdDesc::CreateLayer";
    const auto deconvolution_desc = dynamic_cast<const TrtDeconvolutionDesc*>(layer_desc);
    T_CHECK(deconvolution_desc);

    auto& input = *input_tensors[0];
    nvinfer1::IDeconvolutionLayer* deconvolution = network->addDeconvolutionNd(
        input, deconvolution_desc->nbOutputMaps, deconvolution_desc->kernelSize,
        deconvolution_desc->kernelWeights, deconvolution_desc->biasWeights);

    if (deconvolution == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [deconvolution] layer.";
      return {};
    }

    deconvolution->setStrideNd(deconvolution_desc->stride);
    deconvolution->setPrePadding(deconvolution_desc->prePadding);
    deconvolution->setPostPadding(deconvolution_desc->postPadding);
    deconvolution->setNbGroups(deconvolution_desc->nbGroups);
    return {deconvolution->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
