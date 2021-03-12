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
 * \brief TRT 矩阵乘加层创建器
 */
template <>
class TLayerCreator<TrtMatMulAddDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtMatMulAddDesc::CreateLayer";
    const auto linear_desc = dynamic_cast<const TrtMatMulAddDesc*>(layer_desc);
    T_CHECK(linear_desc);

    nvinfer1::ITensor *input, *weights, *bias;
    input = input_tensors[0];
    weights = input_tensors[1];
    bias = input_tensors[2];

    auto mat_mul = network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE, *weights,
                                              nvinfer1::MatrixOperation::kNONE);
    if (mat_mul == nullptr) {
      LOG(ERROR) << "Failed to add matrix multiply layer";
      LOG(ERROR) << "Create Network: Fail to create [MatMulAdd] layer";
      return {};
    }
    auto mat_mul_output = mat_mul->getOutput(0);

    if (linear_desc->alpha != 1) {
      const int nbDims = mat_mul->getOutput(0)->getDimensions().nbDims;
      const auto alpha_tensor =
          network
              ->addConstant(TrtUtils::CreateSingleton(nbDims),
                            {nvinfer1::DataType::kFLOAT, &linear_desc->alpha, 1})
              ->getOutput(0);
      mat_mul_output = network
                           ->addElementWise(*mat_mul_output, *alpha_tensor,
                                            nvinfer1::ElementWiseOperation::kPROD)
                           ->getOutput(0);
    }

    if (linear_desc->beta != 1) {
      const int nbDims = bias->getDimensions().nbDims;
      const auto beta_tensor =
          network
              ->addConstant(TrtUtils::CreateSingleton(nbDims),
                            {nvinfer1::DataType::kFLOAT, &linear_desc->alpha, 1})
              ->getOutput(0);
      bias = network->addElementWise(*bias, *beta_tensor, nvinfer1::ElementWiseOperation::kPROD)
                 ->getOutput(0);
    }

    auto element_wise =
        network->addElementWise(*mat_mul_output, *bias, nvinfer1::ElementWiseOperation::kSUM);
    if (element_wise == nullptr) {
      LOG(ERROR) << "Failed to add element wise layer";
      LOG(ERROR) << "Create Network: Fail to create [MatMulAdd] layer";
      return {};
    }
    return {element_wise->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
