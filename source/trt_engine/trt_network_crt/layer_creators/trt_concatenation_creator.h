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
 * \brief TRT 拼接层创建器
 */
template <>
class TLayerCreator<TrtConcatenationDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtConcatenationDesc::CreateLayer";
    const auto cat_desc = dynamic_cast<const TrtConcatenationDesc*>(layer_desc);
    T_CHECK(cat_desc);

    nvinfer1::IConcatenationLayer* concat =
        network->addConcatenation(input_tensors.data(), input_tensors.size());

    if (concat == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [concat] layer.";
      return {};
    }

    const int axis = cat_desc->axis;
    concat->setAxis(axis);

    if (cat_desc->is_stack) {
      // TODO(Ao Li): 这里 Stack 必须保证输入的 Shape 都一致
      const int num_inputs = input_tensors.size();
      auto dims = input_tensors[0]->getDimensions();
      dims.nbDims += 1;

      for (int i = dims.nbDims - 1; i > axis; --i) {
        dims.d[i] = dims.d[i - 1];
      }

      // 如果在 batch 维度进行 Stack
      dims.d[axis] = num_inputs;

      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*concat->getOutput(0));
      if (shuffle == nullptr) {
        LOG(ERROR) << "Create Network: Fail to create [concat::shuffle] layer.";
        return {};
      }
      shuffle->setReshapeDimensions(dims);

      return {shuffle->getOutput(0)};
    }

    return {concat->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
