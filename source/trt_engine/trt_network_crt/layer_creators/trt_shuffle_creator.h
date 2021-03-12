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
 * \brief TRT 形状重整层创建器
 */
template <>
class TLayerCreator<TrtShuffleDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtShuffleDesc::CreateLayer";
    const auto shuffle_desc = dynamic_cast<const TrtShuffleDesc*>(layer_desc);
    T_CHECK(shuffle_desc);

    auto& input = *input_tensors[0];
    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(input);

    if (shuffle == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [shuffle] layer.";
      return {};
    }

    if (shuffle_desc->doFirstTrans) {
      shuffle->setFirstTranspose(shuffle_desc->firstTranspose);
    }
    if (shuffle_desc->doReshape) {
      // TODO(Ao Li): 支持 tf ExpandDims
      // for
      //   expand dims with more than 1 unknown dimension auto dim =
      //       shuffle_desc->reshapeDimensions;
      // auto input_dim = input.getDimensions();
      // for (int i = 1; i < dim.nbDims; i++) {
      //   if (dim.d[i] == -1) dim.d[i] = input_dim.d[i];
      // }
      // shuffle->setReshapeDimensions(dim);

      shuffle->setReshapeDimensions(shuffle_desc->reshapeDimensions);
    }
    if (shuffle_desc->doSecondTrans) {
      shuffle->setSecondTranspose(shuffle_desc->secondTranspose);
    }

    return {shuffle->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
