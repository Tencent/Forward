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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN

/**
 * \brief Split 层描述创建器
 */
template <>
class TLayerDescCreator<TrtSplitDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();
    return type == "Split";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtSplitDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const int num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 2);

    const auto input = op.Input(1);
    const nvinfer1::Dims dims = Utils::DimsOf(input);

    op_inputs.push_back(input);

    int split_dim = op.Input(0).GetConstantTensor().AsInt();
    split_dim = (split_dim + dims.nbDims) % dims.nbDims;
    const int64_t num_split = op.GetAttrInt("num_split");

    T_CHECK_LT(split_dim, dims.nbDims);
    T_CHECK_EQ(dims.d[split_dim] % num_split, 0);

    // TODO(yzx): 这里暂时只支持3维输入，根据实际维度进行切分，不进行切分维度转换
    T_CHECK_EQ(dims.nbDims, 3);
    auto layer_desc = std::make_shared<TrtSplitDesc>();
    const int chunk_size = dims.d[split_dim] / num_split;
    layer_desc->dim = (dims.nbDims == 4 ? TrtUtils::NHWC2NCHWDim(split_dim) : split_dim);
    layer_desc->splitSize.resize(num_split, chunk_size);

    for (auto& size : layer_desc->splitSize) {
      auto chunk_dims = dims;
      chunk_dims.d[layer_desc->dim] = size;
      for (int i = 0; i < chunk_dims.nbDims; ++i) {
        if (chunk_dims.d[i] < 0) {
          chunk_dims.d[i] = std::numeric_limits<int>::max();
          layer_desc->dynamic_size = true;
        }
      }

      layer_desc->chunk_sizes.emplace_back(TrtUtils::ToVector(chunk_dims));
    }

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
