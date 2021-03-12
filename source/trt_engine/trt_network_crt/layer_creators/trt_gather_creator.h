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
 * \brief TRT Gather 层创建器
 */
template <>
class TLayerCreator<TrtGatherDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtGatherDesc::CreateLayer";
    const TrtGatherDesc* const gather_desc = dynamic_cast<const TrtGatherDesc*>(layer_desc);
    T_CHECK(gather_desc);

    nvinfer1::ITensor* input{nullptr};
    nvinfer1::ITensor* indices{nullptr};
    // 两个输入都是正常的ITensor*
    if (input_tensors.size() == 2) {
      input = input_tensors[0];
      indices = input_tensors[1];
    } else if (input_tensors.size() == 1) {  // 输入中的一个是常量
      if (gather_desc->inputs[0].inUse) {
        indices = input_tensors[0];
        const nvinfer1::IConstantLayer* input_layer =
            network->addConstant(gather_desc->inputs[0].dim,
                                 {nvinfer1::DataType::kFLOAT, gather_desc->inputs[0].data.Data(),
                                  gather_desc->inputs[0].data.Count()});
        input = input_layer->getOutput(0);
      } else {
        input = input_tensors[0];
        const nvinfer1::IConstantLayer* indices_layer =
            network->addConstant(gather_desc->inputs[1].dim,
                                 {nvinfer1::DataType::kFLOAT, gather_desc->inputs[1].data.Data(),
                                  gather_desc->inputs[1].data.Count()});
        indices = indices_layer->getOutput(0);
      }
    }

    const auto axis = gather_desc->gatherAxis;

    nvinfer1::IGatherLayer* gather_layer = network->addGather(*input, *indices, axis);

    if (!gather_layer) {
      LOG(ERROR) << "Create Network: Fail to create [gather] layer";
      return {};
    }

    auto* output = gather_layer->getOutput(0);

    return {output};
  }
};

FWD_TRT_NAMESPACE_END
