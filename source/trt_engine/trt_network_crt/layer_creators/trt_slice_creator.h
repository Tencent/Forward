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
 * \brief TRT 切片层创建器
 */
template <>
class TLayerCreator<TrtSliceDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtSliceDesc::CreateLayer";
    const auto slice_desc = dynamic_cast<const TrtSliceDesc*>(layer_desc);
    T_CHECK(slice_desc);

    auto& input = *input_tensors[0];

    // so far, assume stride is positive
    nvinfer1::ISliceLayer* slice =
        network->addSlice(input, slice_desc->start, slice_desc->size, slice_desc->stride);

    if (slice_desc->dynamic_size || slice_desc->dynamic_start) {
      auto shape = network->addShape(input)->getOutput(0);

      if (slice_desc->dynamic_start) {
        auto start_t = CreateStartTensor(network, slice_desc, shape);
        slice->setInput(1, *start_t);
      }

      if (slice_desc->dynamic_size) {
        auto size_t = CreateSizeTensor(network, slice_desc, shape);
        slice->setInput(2, *size_t);
      }
    }

    if (slice == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [slice] layer.";
      return {};
    }

    return {slice->getOutput(0)};
  }

 private:
  nvinfer1::ITensor* CreateStartTensor(nvinfer1::INetworkDefinition* network,
                                       const TrtSliceDesc* slice_desc, nvinfer1::ITensor* shape) {
    // final_start = min(input_shape, start_params_positive) +
    // (start_params_negative) where negative value becomes INT_MAX in the
    // start_params_positive
    const nvinfer1::Dims shape_dims = {1, slice_desc->start.nbDims};

    const auto start_vec_raw =
        network->addConstant(shape_dims, slice_desc->weight_map.at("start_raw"))->getOutput(0);

    const auto start_tmp =
        network->addElementWise(*start_vec_raw, *shape, nvinfer1::ElementWiseOperation::kMIN)
            ->getOutput(0);

    const auto diff_t =
        network->addConstant(shape_dims, slice_desc->weight_map.at("start_negative"))->getOutput(0);

    return network->addElementWise(*start_tmp, *diff_t, nvinfer1::ElementWiseOperation::kSUM)
        ->getOutput(0);
  }

  nvinfer1::ITensor* CreateSizeTensor(nvinfer1::INetworkDefinition* network,
                                      const TrtSliceDesc* slice_desc, nvinfer1::ITensor* shape) {
    // final_size = min(input_shape, start_params)
    const nvinfer1::Dims shape_dims = {1, slice_desc->size.nbDims};
    const auto size_vec_t =
        network->addConstant(shape_dims, slice_desc->weight_map.at("size"))->getOutput(0);

    return network->addElementWise(*size_vec_t, *shape, nvinfer1::ElementWiseOperation::kMIN)
        ->getOutput(0);
  }
};

FWD_TRT_NAMESPACE_END
