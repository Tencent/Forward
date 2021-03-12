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
 * \brief TRT Clamp 层创建器（通过两个ElementWise层实现）
 */
template <>
class TLayerCreator<TrtClampDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtClampDesc::CreateLayer";

    const auto clamp_desc = dynamic_cast<const TrtClampDesc*>(layer_desc);
    T_CHECK(clamp_desc);

    auto input_tensor = input_tensors[0];

    if (clamp_desc->has_min) {
      input_tensor = CreateElementWise(network, &clamp_desc->min, *input_tensor,
                                       nvinfer1::ElementWiseOperation::kMAX);
      if (input_tensor == nullptr) {
        LOG(ERROR) << "Create Network: Fail to create [Clamp] layer.";
        return {};
      }
    }

    if (clamp_desc->has_max) {
      input_tensor = CreateElementWise(network, &clamp_desc->max, *input_tensor,
                                       nvinfer1::ElementWiseOperation::kMIN);
      if (input_tensor == nullptr) {
        LOG(ERROR) << "Create Network: Fail to create [Clamp] layer.";
        return {};
      }
    }

    return {input_tensor};
  }

 private:
  nvinfer1::ITensor* CreateElementWise(nvinfer1::INetworkDefinition* network, const float* value,
                                       nvinfer1::ITensor& input,
                                       nvinfer1::ElementWiseOperation op) const {
    const nvinfer1::Dims dims{input.getDimensions().nbDims, {1, 1, 1, 1, 1, 1, 1, 1}};

    const auto constant = network->addConstant(dims, {nvinfer1::DataType::kFLOAT, value, 1});
    if (constant == nullptr) {
      LOG(ERROR) << "Failed to add constant layer";
      return nullptr;
    }

    const auto element_wise = network->addElementWise(input, *constant->getOutput(0), op);
    if (element_wise == nullptr) {
      LOG(ERROR) << "Failed to add element wise layer";
      return nullptr;
    }

    return element_wise->getOutput(0);
  }
};

FWD_TRT_NAMESPACE_END
