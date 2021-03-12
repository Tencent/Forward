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
 * \brief TRT 自仿层创建器
 */
template <>
class TLayerCreator<TrtIdentityDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtIdentityDesc::CreateLayer";
    const auto identity_desc = dynamic_cast<const TrtIdentityDesc*>(layer_desc);
    T_CHECK(identity_desc);

    if (!input_tensors.empty()) {
      if (!identity_desc->copy) {
        return input_tensors;
      }

      ITensorVector outputs;
      for (auto* input : input_tensors) {
        // kINT32 is not valid input. To avoid error, just return the input.
        if (input->getType() == nvinfer1::DataType::kINT32) {
          LOG(WARNING) << "kINT32 value cannot be copied by identity layer. "
                          "Continue with raw tensor.";
          outputs.push_back(input);
        } else {
          auto identity = network->addIdentity(*input);
          if (identity == nullptr) {
            LOG(ERROR) << "Create Network: Fail to create [identity] layer.";
            return {};
          }
          outputs.push_back(identity->getOutput(0));
        }
      }
      return outputs;
    }

    // 常量 Identity 处理
    if (!identity_desc->input.inUse) {
      return {};
    }

    int size = 1;
    for (int i = 0; i < identity_desc->input.dim.nbDims; i++) {
      size *= identity_desc->input.dim.d[i];
    }
    auto identity = network->addConstant(identity_desc->input.dim,
                                         nvinfer1::Weights(identity_desc->input.data));
    if (identity == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [identity] layer.";
      return {};
    }
    return {identity->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
