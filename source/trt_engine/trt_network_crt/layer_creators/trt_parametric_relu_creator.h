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
 * \brief TRT Parametric ReLU层创建器
 */
template <>
class TLayerCreator<TrtParametricReLUDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtParametricReLUDesc::CreateLayer";
    const auto prelu_desc = dynamic_cast<const TrtParametricReLUDesc*>(layer_desc);
    T_CHECK(prelu_desc);
    // TODO(Ao Li): 暂时以 Constant 作为第二个参数，Eval 模式下是否有可能不是
    // Constant ?
    nvinfer1::IConstantLayer* slop =
        network->addConstant(prelu_desc->slopDims, prelu_desc->slopWeights);

    if (slop == nullptr) {
      LOG(ERROR) << "Failed to add constant layer.";
      LOG(ERROR) << "Create Network: Fail to create [PReLU] layer.";
      return {};
    }

    nvinfer1::IParametricReLULayer* prelu =
        network->addParametricReLU(*input_tensors[0], *slop->getOutput(0));

    if (prelu == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [PReLU] layer.";
      return {};
    }
    return {prelu->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
