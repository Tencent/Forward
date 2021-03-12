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
 * \brief TRT 全连接层创建器
 */
template <>
class TLayerCreator<TrtFullyConnectedDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtFullyConnectedDesc::CreateLayer";
    const auto fully_connected_desc = dynamic_cast<const TrtFullyConnectedDesc*>(layer_desc);
    T_CHECK(fully_connected_desc);

    auto* input = input_tensors[0];

    // FC层只接受non_batch dim >= 3的输入(然后内部实现flatten)
    // 输入的后三维会被自动flatten，batch维保持不变，其他维度会被flatten.
    // 对于batch*N*M或batch*M的矩阵，为保持正常的乘一个M*K的矩阵，需要手动添加维度到n+2维
    auto dims = input->getDimensions();
    bool add_dim = false;
    if (dims.nbDims <= 3) {
      add_dim = true;
      const auto shuffle = network->addShuffle(*input);
#ifdef USE_DYNAMIC_BATCH
      dims.d[0] = 0;
      const auto kv = fully_connected_desc->kernelWeights.Dims();
      if (dims.nbDims == 2) dims.d[1] = kv.d[1];
      if (dims.nbDims == 3) {
        dims.d[2] = kv.d[1];
      }
#endif
      dims.d[dims.nbDims++] = 1;
      dims.d[dims.nbDims++] = 1;
      shuffle->setReshapeDimensions(dims);
      input = shuffle->getOutput(0);
    }

    nvinfer1::IFullyConnectedLayer* fully_connected = network->addFullyConnected(
        *input, fully_connected_desc->nbOutputChannels, fully_connected_desc->kernelWeights,
        fully_connected_desc->biasWeights);

    if (!fully_connected) {
      LOG(ERROR) << "Create Network: Fail to create [fully connected] layer";
      return {};
    }

    auto* output = fully_connected->getOutput(0);
    if (add_dim) {
      const auto shuffle = network->addShuffle(*output);
      dims = output->getDimensions();
#ifdef USE_DYNAMIC_BATCH
      for (int i = 0; i < dims.nbDims; ++i) dims.d[i] = 0;
#endif
      dims.nbDims -= 2;
      shuffle->setReshapeDimensions(dims);
      output = shuffle->getOutput(0);
    }
    return {output};
  }
};

FWD_TRT_NAMESPACE_END
