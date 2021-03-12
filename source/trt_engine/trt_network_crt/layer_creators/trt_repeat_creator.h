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
 * \brief TRT Repeat 层创建器
 */
template <>
class TLayerCreator<TrtRepeatDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtRepeatDesc::CreateLayer";
    const TrtRepeatDesc* const repeat_desc = dynamic_cast<const TrtRepeatDesc*>(layer_desc);
    T_CHECK(repeat_desc);

    const auto input_dims = input_tensors[0]->getDimensions();
    const auto& repeats = repeat_desc->repeats;

    // repeats size 必须不小于输入维度
    const int delta = repeats.size() - input_dims.nbDims;
    T_CHECK_GE(delta, 0);

    nvinfer1::ITensor* output{input_tensors[0]};

    nvinfer1::Dims output_dims;
    output_dims.nbDims = static_cast<int>(repeats.size());

    // 前置维度如果不是1，那么全都叠加在0维度上
    int before_dims = 1;
    for (int i = 0; i < delta; ++i) {
      before_dims *= repeats[i];
      output_dims.d[i] = repeats[i];
    }
    ITensorVector temp;
    if (before_dims != 1) {
      temp.assign(before_dims, output);
      output = AddConcatenation(network, temp, 0);
    }

    // 现在处理对应的维度
    for (int i = 0; i < input_dims.nbDims; ++i) {
      T_CHECK_GE(repeats[delta + i], 1);
      if (repeats[delta + i] > 1) {
        temp.assign(repeats[delta + i], output);
        output = AddConcatenation(network, temp, i);
      }
      output_dims.d[delta + i] = repeats[delta + i] * input_dims.d[i];
    }

    // 进行最后的 reshape 操作
    // TODO(yzx): 此处唯一的 -1 的值可被 TRT 推导出来，但是因为 Engine 的
    // Output Dimension 暂不支持动态指定，所以 repeat 目前无法通过单元测试。
    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*output);
    for (int i = 0; i < output_dims.nbDims; ++i) {
      if (output_dims.d[i] < -1) output_dims.d[i] = -1;
    }
    shuffle->setReshapeDimensions(output_dims);
    return {shuffle->getOutput(0)};
  }

 private:
  static nvinfer1::ITensor* AddConcatenation(nvinfer1::INetworkDefinition* network,
                                             const ITensorVector& inputs, int axis) {
    nvinfer1::IConcatenationLayer* concat =
        network->addConcatenation(inputs.data(), static_cast<int>(inputs.size()));
    concat->setAxis(axis);
    return concat->getOutput(0);
  }
};

FWD_TRT_NAMESPACE_END
