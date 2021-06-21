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

#include <memory>
#include <string>
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief Shuffle 层描述创建器
 */
template <>
class TLayerDescCreator<TrtShuffleDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();
    return type == "Transpose" || type == "Reshape" || type == "ExpandDims" || type == "Squeeze";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtShuffleDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    auto layer_desc = std::make_shared<TrtShuffleDesc>();

    const auto type = op.OpType();
    if (type == "Transpose") {
      T_CHECK_EQ(op.NumInputs(), 2);

      // Input 0, kind = input
      // Input 1, perm
      const auto input = op.Input(0);
      const auto perm = op.Input(1);

      // 将输入返回
      op_inputs.push_back(input);

      // [N, H, W, C]
      auto const_permute = perm.GetConstantTensor();
      T_CHECK(const_permute.Valid());
      const std::vector<int> permutes = const_permute.AsIntList();
      T_CHECK_EQ(permutes.size(), 4);

      layer_desc->doFirstTrans = true;
      layer_desc->doReshape = false;
      layer_desc->doSecondTrans = false;
      layer_desc->firstTranspose =
          TrtUtils::ToPermutation(TrtUtils::NHWC2NCHWDim(TrtUtils::NHWC2NCHW(permutes)));
    } else if (type == "Reshape") {
      T_CHECK_EQ(op.NumInputs(), 2);

      // Input 0, kind = input
      // Input 1, shape
      const auto input = op.Input(0);
      const auto reshape = op.Input(1);

      // 将输入返回
      op_inputs.push_back(input);
      std::vector<int> dims;
      if (reshape.OpType() == "Shape") {
        std::vector<int64_t> dims_64;
        const auto tensor_to_get_shape = reshape.Input(0);
        dims_64.resize(tensor_to_get_shape.GetTensorNumDims());
        tensor_to_get_shape.GetTensorShape(dims_64.data(), dims_64.size());
        dims.assign(dims_64.begin(), dims_64.end());
      } else if (reshape.OpType() == "Pack") {
        // keep batch dim
        dims.push_back(0);
        // assign other params in Pack Node
        for (int i = 1; i < reshape.NumInputs(); ++i) {
          auto reshape_dim = reshape.Input(i).GetConstantTensor();
          T_CHECK(reshape_dim.Valid());
          dims.push_back(reshape_dim.AsInt());
        }
      } else {
        auto reshape_dims = reshape.GetConstantTensor();
        T_CHECK(reshape_dims.Valid());
        dims = reshape_dims.AsIntList();
      }
      layer_desc->doReshape = true;
      layer_desc->reshapeDimensions = TrtUtils::ToDims(TrtUtils::NHWC2NCHW(dims));

      // When inputs are NHWC, we do NHWC->NCHW on those inputs to match TensorRT's data format.
      // If the number dimensions of Reshape changed, we have to do NCHW->NHWC to keep
      // data storages correct for following layers.
      if (input.GetTensorNumDims() == 4 && dims.size() != 4) {
        layer_desc->doFirstTrans = true;
        layer_desc->firstTranspose = {0, 2, 3, 1, 0, 0, 0, 0};
      } else {
        layer_desc->doFirstTrans = false;
      }
      layer_desc->doSecondTrans = false;
    } else if (type == "ExpandDims") {
      T_CHECK_EQ(op.NumInputs(), 2);

      // Input 0, kind = input
      // Input 1, perm
      const auto input = op.Input(0);
      const auto dim = op.Input(1);

      auto dims = DimsOf(input);
      dims.nbDims += 1;

      // int expand_dim = GetConstantInt(graph, dim);
      auto const_dim = dim.GetConstantTensor();
      T_CHECK(const_dim.Valid());
      int expand_dim = const_dim.AsInt();
      if (expand_dim < 0) {
        expand_dim += dims.nbDims;
      }
      for (int i = dims.nbDims - 1; i >= expand_dim + 1; i--) {
        dims.d[i] = dims.d[i - 1];
      }
      dims.d[expand_dim] = 1;

      layer_desc->doFirstTrans = false;
      layer_desc->doReshape = true;
      layer_desc->doSecondTrans = false;
      layer_desc->reshapeDimensions = dims;

      // 将输入返回
      op_inputs.push_back(input);
    } else if (type == "Squeeze") {
      T_CHECK_EQ(op.NumInputs(), 1);

      // Input 0, kind = input
      const auto input = op.Input(0);
      const auto squeeze_dim = op.GetAttrIntList("squeeze_dims");

      const auto dims = DimsOf(input);

      nvinfer1::Dims real_dims;
      real_dims.nbDims = 0;

      for (int i = 0; i < dims.nbDims; ++i) {
        if (std::find(squeeze_dim.begin(), squeeze_dim.end(), i) == squeeze_dim.end()) {
          real_dims.d[real_dims.nbDims++] = dims.d[i];
        }
      }

      layer_desc->doFirstTrans = false;
      layer_desc->doReshape = true;
      layer_desc->doSecondTrans = false;
      layer_desc->reshapeDimensions = real_dims;

      // 将输入返回
      op_inputs.push_back(input);
    }

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
