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
 * \brief Pooling 池化层描述创建器
 */
template <>
class TLayerDescCreator<TrtPoolingDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const std::string type = op.OpType();

    return type == "MaxPool" || type == "AvgPool";
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtPoolingDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    const auto num_inputs = op.NumInputs();
    T_CHECK_EQ(num_inputs, 1);

    // Input 0, kind = input
    const auto input = op.Input(0);

    // 将输入返回
    op_inputs.push_back(input);

    Status status;

    // 获取属性

    // 1. kernel size
    // 2. padding
    // 3. type
    // 4. strides
    // 5. data format
    std::vector<int64_t> ksize = op.GetAttrIntList("ksize");
    const std::string padding = op.GetAttrString("padding");
    const TF_DataType type = op.GetAttrType("T");
    std::vector<int64_t> strides = op.GetAttrIntList("strides");
    const std::string data_format = op.GetAttrString("data_format");

    const std::string pooling_type = op.OpType();

    auto layer_desc = std::make_shared<TrtPoolingDesc>();

    if (pooling_type == "MaxPool") {
      layer_desc->poolingType = nvinfer1::PoolingType::kMAX;
    } else if (pooling_type == "AvgPool") {
      layer_desc->poolingType = nvinfer1::PoolingType::kAVERAGE;
    } else {
      LOG(ERROR) << "Unsupported pooling type " << pooling_type;
      return nullptr;
    }

    // TODO(Ao Li): 目前先支持 pooling2D
    if (data_format == "NHWC") {
      ksize = TrtUtils::NHWC2NCHW(ksize);
      strides = TrtUtils::NHWC2NCHW(strides);
    } else if (data_format != "NCHW") {
      LOG(ERROR) << "Unsupported data format " << data_format;
      return nullptr;
    }

    T_CHECK(ksize[0] == 1 && ksize[1] == 1);
    T_CHECK(strides[0] == 1 && strides[1] == 1);

    layer_desc->windowSize = TrtUtils::ToDims(std::vector<int64_t>{ksize[2], ksize[3]});
    layer_desc->stride = TrtUtils::ToDims(std::vector<int64_t>{strides[2], strides[3]});

    if (padding == "SAME") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } else if (padding == "VALID") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    } else {
      LOG(ERROR) << "Unsupported padding mode " << padding;
      return nullptr;
    }

    // zero-padded
    layer_desc->padding.nbDims = 0;
    layer_desc->averageCountExcludesPadding = true;

    return layer_desc;
  }
};

FWD_TF_NAMESPACE_END
