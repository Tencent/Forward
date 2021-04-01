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

// TRT Slice Description Creator
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

    SetDynamicInput(network, slice_desc, input, slice);

    if (slice == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [slice] layer.";
      return {};
    }

    return {slice->getOutput(0)};
  }

 private:
  void SetDynamicInput(nvinfer1::INetworkDefinition* network, const TrtSliceDesc* const slice_desc,
                       nvinfer1::ITensor& input, nvinfer1::ISliceLayer* slice) {
    if (!CreateDynamicParameters(slice_desc)) return;

    const auto shape = network->addShape(input)->getOutput(0);

    if (slice_desc->dynamic_start) {
      const auto start_t = CreateDynamicParam(network, shape, slice_desc->weight_map.at("start"),
                                              slice_desc->weight_map.at("start_negative"));
      slice->setInput(1, *start_t);
    }

    const auto size_t = CreateDynamicParam(network, shape, slice_desc->weight_map.at("size"),
                                           slice_desc->weight_map.at("size_negative"));
    slice->setInput(2, *size_t);
  }

  bool CreateDynamicParameters(const TrtSliceDesc* slice_desc) {
    auto m_slice_desc = const_cast<TrtSliceDesc*>(slice_desc);

    // start dynamic
    auto start_vec = TrtUtils::ToVector(m_slice_desc->start);
    auto start_neg_vec = start_vec;
    int start_has_neg = 0;
    for (auto& s : start_vec) {
      start_has_neg |= s;
      s = s < 0 ? std::numeric_limits<int>::max() : s;
    }
    if (start_has_neg < 0) {
      m_slice_desc->dynamic_start = true;
      m_slice_desc->weight_map["start"] = FwdWeights(start_vec);
      for (auto& s : start_neg_vec) s = s < 0 ? s : 0;
      m_slice_desc->weight_map["start_negative"] = FwdWeights(start_neg_vec);
    }

    // size dynamic
    auto size_vec = TrtUtils::ToVector(m_slice_desc->size);
    auto size_neg_vec = size_vec;
    int size_has_neg = 0;
    for (auto& s : size_vec) {
      size_has_neg |= s;
      // here use INT_MAX instead of negative value
      s = s < 0 ? std::numeric_limits<int>::max() : s;
    }
    if (m_slice_desc->dynamic_start || size_has_neg < 0) {
      m_slice_desc->dynamic_size = true;
      m_slice_desc->weight_map["size"] = FwdWeights(size_vec);
      for (auto& s : size_neg_vec) s = s < 0 ? s + 1 : 0;
      m_slice_desc->weight_map["size_negative"] = FwdWeights(size_neg_vec);
    }

    return m_slice_desc->dynamic_size || m_slice_desc->dynamic_start;
  }

  nvinfer1::ITensor* CreateDynamicParam(nvinfer1::INetworkDefinition* network,
                                        nvinfer1::ITensor* shape, const FwdWeights& param_pos,
                                        const FwdWeights& param_neg) {
    // final_start = min(input_shape, start_params_positive) +
    // (start_params_negative) where negative value becomes INT_MAX in the
    // start_params_positive

    const nvinfer1::Dims shape_dims = {1, static_cast<int>(param_pos.Count())};

    const auto start_vec_raw = network->addConstant(shape_dims, param_pos)->getOutput(0);

    const auto start_tmp =
        network->addElementWise(*start_vec_raw, *shape, nvinfer1::ElementWiseOperation::kMIN)
            ->getOutput(0);

    const auto diff_t = network->addConstant(shape_dims, param_neg)->getOutput(0);

    return network->addElementWise(*start_tmp, *diff_t, nvinfer1::ElementWiseOperation::kSUM)
        ->getOutput(0);
  }
};

FWD_TRT_NAMESPACE_END
