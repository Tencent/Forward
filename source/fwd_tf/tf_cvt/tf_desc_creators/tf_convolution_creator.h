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
 * \brief Convolution 卷积层描述创建器
 */
template <>
class TLayerDescCreator<TrtConvolutionDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    if (CheckConv1D(op)) return true;

    const auto type = op.OpType();
    if (type == "Conv2D" || type == "DepthwiseConv2dNative") {
      has_bias_ = false;
      return true;
    }

    if (type == "BiasAdd") {
      const auto input0_type = op.Input(0).OpType();

      if (input0_type == "Conv2D" || input0_type == "DepthwiseConv2dNative") {
        has_bias_ = true;
        return true;
      }

      // Conv1D
      if (CheckConv1D(op.Input(0))) {
        has_bias_ = true;
        return true;
      }
    }

    return false;
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& oper, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtConvolutionDesc::Create";
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;
    T_CHECK_EQ(oper.NumInputs(), 2);

    Operation op = oper;

    auto layer_desc = std::make_shared<TrtConvolutionDesc>();

    if (has_bias_) {
      // Input 0, kind = input
      // Input 1, kind = bias
      const auto input = op.Input(0);
      const auto bias = op.Input(1);

      // 属性
      const TF_DataType type = op.GetAttrType("T");
      const std::string data_format = op.GetAttrString("data_format");
      T_CHECK_EQ(data_format, "NHWC");

      layer_desc->biasWeights = ToFwdWeights(bias.GetConstantTensor());

      // op 转移到卷积层
      op = input;
    }

    // Conv1D 需要跳过 Squeeze 层
    if (conv_1d_) {
      T_CHECK_EQ(op.OpType(), "Squeeze");
      const auto squeeze_dims = op.GetAttrIntList("squeeze_dims");
      T_CHECK_EQ(squeeze_dims.size(), 1);
      layer_desc->squeeze_dim = TrtUtils::NHWC2NCHWDim(squeeze_dims[0]);
      op = op.Input(0);
    }

    // Input 0, kind = input
    // Input 1, kind = filter
    const auto input = op.Input(0);
    const auto filter = op.Input(1);

    // 将输入返回
    op_inputs.push_back(input);

    const std::string op_type = op.OpType();

    // 获取属性
    std::vector<int64_t> dilations = op.GetAttrIntList("dilations");
    const TF_DataType type = op.GetAttrType("T");
    const std::string data_format = op.GetAttrString("data_format");
    std::vector<int64_t> strides = op.GetAttrIntList("strides");
    std::vector<int64_t> explicit_paddings;
    const std::string padding = op.GetAttrString("padding");

    T_CHECK_EQ(data_format, "NHWC");

    // tensorflow 卷积核维度为 [kH, kW, in_c, out_c]
    // 而 TensorRT 卷积核参数维度为  [out_c, in_c, kH, kW], 因此这里需要
    // (3,2,0,1) 转置
    const auto filter_dims = DimsOf(filter);
    T_CHECK_EQ(filter_dims.nbDims, 4);
    layer_desc->kernelSize = TrtUtils::ToDims(std::vector<int>{filter_dims.d[0], filter_dims.d[1]});
    layer_desc->kernelWeights = ToFwdWeights(filter.GetConstantTensor());
    layer_desc->kernelWeights.Transpose(filter_dims, {3, 2, 0, 1});

    if (op_type == "DepthwiseConv2dNative") {
      layer_desc->nbOutputMaps = filter_dims.d[2] * filter_dims.d[3];
      layer_desc->nbGroups = filter_dims.d[2];
    } else {
      layer_desc->nbOutputMaps = filter_dims.d[3];
      explicit_paddings = op.GetAttrIntList("explicit_paddings");
      layer_desc->nbGroups = 1;
    }

    dilations = TrtUtils::NHWC2NCHW(dilations);
    strides = TrtUtils::NHWC2NCHW(strides);
    explicit_paddings = TrtUtils::NHWC2NCHW(explicit_paddings);

    // TODO(Ao Li): 目前先支持 Conv2D
    T_CHECK(strides[0] == 1 && strides[1] == 1);
    T_CHECK(dilations[0] == 1 && dilations[1] == 1);

    layer_desc->stride = TrtUtils::ToDims(std::vector<int64_t>{strides[2], strides[3]});
    layer_desc->dilation = TrtUtils::ToDims(std::vector<int64_t>{dilations[2], dilations[3]});

    // TODO(Ao Li): 处理 explicit_paddings 不为空的情况
    T_CHECK(explicit_paddings.empty());
    layer_desc->prePadding.nbDims = 0;
    layer_desc->postPadding.nbDims = 0;

    if (padding == "SAME") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } else if (padding == "VALID") {
      layer_desc->paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    } else {
      LOG(ERROR) << "Unsupported padding mode " << padding;
      return nullptr;
    }

    return layer_desc;
  }

 private:
  bool has_bias_{false};

  bool conv_1d_{false};

  // 识别 Conv1d 的情况，这里认为以下模式是 Conv1d
  // ------------              ------------
  // |ExpandDims|              |ExpandDims|
  // ------------              ------------
  //      |                          |
  //      ----------------------------
  //                   |
  //                --------
  //                |Conv2D|
  //                --------
  //                   |
  //               ---------
  //               |Squeeze|
  //               ---------
  // 后接可选的 BiasAdd
  bool CheckConv1D(const Operation& op) {
    const auto type = op.OpType();
    if (type == "Squeeze") {
      const auto child = op.Input(0);
      if (child.OpType() == "Conv2D" && child.Input(0).OpType() == "ExpandDims" &&
          child.Input(1).OpType() == "ExpandDims") {
        conv_1d_ = true;
        return true;
      }
    }
    return false;
  }
};

FWD_TF_NAMESPACE_END
