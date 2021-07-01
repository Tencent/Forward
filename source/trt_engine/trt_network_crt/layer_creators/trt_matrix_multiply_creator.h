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
 * \brief TRT 矩阵乘法层创建器
 */
template <>
class TLayerCreator<TrtMatrixMultiplyDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtMatrixMultiplyDesc::CreateLayer";
    const auto mat_mul_desc = dynamic_cast<const TrtMatrixMultiplyDesc*>(layer_desc);
    T_CHECK(mat_mul_desc);

    nvinfer1::ITensor* input0{nullptr};
    nvinfer1::ITensor* input1{nullptr};
    // 两个输入都是正常的ITensor*
    if (input_tensors.size() == 2) {
      input0 = input_tensors[0];
      input1 = input_tensors[1];
    } else if (input_tensors.size() == 1) {  // 输入中的一个是常量
      if (mat_mul_desc->inputs[0].inUse) {
        input1 = input_tensors[0];
        const nvinfer1::IConstantLayer* input0_layer =
            network->addConstant(mat_mul_desc->inputs[0].dim, mat_mul_desc->inputs[0].data);
        input0 = input0_layer->getOutput(0);
      } else {
        input0 = input_tensors[0];
        const nvinfer1::IConstantLayer* input1_layer =
            network->addConstant(mat_mul_desc->inputs[1].dim, mat_mul_desc->inputs[1].data);
        input1 = input1_layer->getOutput(0);
      }
    }

    T_CHECK_NOTNULL(input0);
    T_CHECK_NOTNULL(input1);
    CHECK_EQ(input0->getDimensions().nbDims, input1->getDimensions().nbDims);

    nvinfer1::IMatrixMultiplyLayer* mat_mul =
        network->addMatrixMultiply(*input0, mat_mul_desc->op[0], *input1, mat_mul_desc->op[1]);
    if (mat_mul == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [matrix multiply] layer";
      return {};
    }
    return {mat_mul->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
