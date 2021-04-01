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
 * \brief Slice 切片层描述创建器
 */
template <>
class TLayerDescCreator<TrtSliceDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const auto type = op.OpType();
    return type == "StridedSlice";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtSliceDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const int num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 4);

    const auto input = op.Input(0);
    op_inputs.push_back(input);

    const auto begin_op = op.Input(1);
    const auto end_op = op.Input(2);
    const auto stride_op = op.Input(3);

    const auto begin_mask = op.GetAttrInt("begin_mask");
    const auto end_mask = op.GetAttrInt("end_mask");
    const auto shrink_axis_mask = op.GetAttrInt("shrink_axis_mask");
    const auto ellipsis_mask = op.GetAttrInt("ellipsis_mask");
    const auto new_axis_mask = op.GetAttrInt("new_axis_mask");

    // TODO(Ao Li): 需要考虑以下 mask 不为 0 的情况
    // T_CHECK_EQ(shrink_axis_mask, 0);
    T_CHECK_EQ(ellipsis_mask, 0);
    T_CHECK_EQ(new_axis_mask, 0);

    auto begin_list = begin_op.GetConstantTensor().AsIntList();
    auto end_list = end_op.GetConstantTensor().AsIntList();
    auto strides_list = stride_op.GetConstantTensor().AsIntList();

    auto layer_desc = std::make_shared<TrtSliceDesc>();
    auto dims = DimsOf(input);

    for (int i = 0; i < dims.nbDims; ++i) {
      if (begin_mask & (1ll << i)) begin_list[i] = 0;
      if (end_mask & (1ll << i)) end_list[i] = dims.d[i];
    }

    layer_desc->start = TrtUtils::ToDims(TrtUtils::NHWC2NCHW(begin_list));
    layer_desc->stride = TrtUtils::ToDims(TrtUtils::NHWC2NCHW(strides_list));

    for (size_t i = 0; i < end_list.size(); ++i) {
      const int start = begin_list[i];
      const int end = end_list[i];
      const int stride = strides_list[i];
      end_list[i] = end < 0 ? (dims.d[i] + end - start + stride - 1) / stride
                            : (end - start + stride - 1) / stride;
    }
    layer_desc->size = TrtUtils::ToDims(TrtUtils::NHWC2NCHW(end_list));

    // batch dim should be -1;
    layer_desc->size.d[0] = -1;

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
