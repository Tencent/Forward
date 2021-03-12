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

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Pooling 层描述创建器
 */
template <>
class TLayerDescCreator<TrtPoolingDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();

    // TODO(Ao Li): 增加对 1d 的支持

    // #define ALWAYS_USE_ADAPTIVE_PLUGIN

    // 这里对特殊情况使用 pooling 直接处理
#ifndef ALWAYS_USE_ADAPTIVE_PLUGIN
    if (kind == c10::aten::adaptive_avg_pool2d || kind == c10::aten::adaptive_avg_pool3d ||
        kind == c10::aten::adaptive_max_pool2d || kind == c10::aten::adaptive_max_pool3d) {
      bool fine = true;
      const at::Tensor dummy = module.Get(node->inputs()[0]).toTensor();
      const auto output_size = module.Get(node->inputs()[1]).toIntList();
      for (int64_t i = 2; i < dummy.ndimension(); ++i) {
        if (dummy.size(i) % output_size[i - 2] != 0) {
          fine = false;
          break;
        }
      }
      return fine;
    }
#endif

    return kind == c10::aten::avg_pool2d || kind == c10::aten::avg_pool3d ||
           kind == c10::aten::max_pool2d || kind == c10::aten::max_pool3d;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtPoolingDesc::Create";

    const auto kind = node->kind();

    if (kind == c10::aten::avg_pool2d || kind == c10::aten::avg_pool3d ||
        kind == c10::aten::max_pool2d || kind == c10::aten::max_pool3d) {
      return CreatePooling(node, module, input_values);
    }

    if (kind == c10::aten::adaptive_max_pool2d || kind == c10::aten::adaptive_max_pool3d ||
        kind == c10::aten::adaptive_avg_pool2d || kind == c10::aten::adaptive_avg_pool3d) {
      return CreateAdaptivePooling(node, module, input_values);
    }

    return nullptr;
  }

 private:
  /**
   * \brief 创建 Pooling 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreatePooling(const JitNode* node, const TorchModule& module,
                                              std::vector<const JitValue*>& input_values) const {
    auto layer_desc = std::make_shared<TrtPoolingDesc>();

    const auto kind = node->kind();

    const auto inputs = node->inputs();

    // Input 0, kind = input
    // Input 1, kind = c10::prim::ListConstruct, kernel_size
    // Input 2, kind = c10::prim::ListConstruct, stride
    // Input 3, kind = c10::prim::ListConstruct, padding
    // Input 4, kind = c10::prim::ListConstruct, dilation/ceil_mode
    // Input 5, kind = c10::prim::Constant, ceil_mode/CountExcludesPadding
    input_values.push_back(inputs[0]);

    if (kind == c10::aten::avg_pool2d || kind == c10::aten::avg_pool3d) {
      T_CHECK_EQ(inputs.size(), 7);
      layer_desc->poolingType = nvinfer1::PoolingType::kAVERAGE;
      layer_desc->paddingMode = nvinfer1::PaddingMode(module.Get(inputs[4]).toBool());
      layer_desc->averageCountExcludesPadding = !module.Get(inputs[5]).toBool();
    } else if (kind == c10::aten::max_pool2d || kind == c10::aten::max_pool3d) {
      T_CHECK_EQ(inputs.size(), 6);
      layer_desc->paddingMode = nvinfer1::PaddingMode(module.Get(inputs[5]).toBool());
      layer_desc->poolingType = nvinfer1::PoolingType::kMAX;
    } else {
      return layer_desc;
    }

    const auto kernel_size = module.Get(inputs[1]).toIntList();
    const auto stride = module.Get(inputs[2]).toIntList();
    const auto padding = module.Get(inputs[3]).toIntList();

    layer_desc->windowSize = Utils::ToDims(kernel_size);

    if (!stride.empty()) {
      layer_desc->stride = Utils::ToDims(stride);
    }

    if (padding.size() == 1 && padding[0] == 0) {
      layer_desc->padding = nvinfer1::Dims({0});
    } else {
      layer_desc->padding = Utils::ToDims(padding);
    }

    return layer_desc;
  }

  /**
   * \brief 创建 AdaptivePooling 层描述
   * \param node
   * \param module
   * \param input_values
   * \return
   */
  std::shared_ptr<TrtLayerDesc> CreateAdaptivePooling(
      const JitNode* node, const TorchModule& module,
      std::vector<const JitValue*>& input_values) const {
    auto layer_desc = std::make_shared<TrtPoolingDesc>();

    const auto kind = node->kind();

    const auto inputs = node->inputs();

    input_values.push_back(inputs[0]);

    // Input 0, kind = input
    // Input 1, kind = c10::prim::ListConstruct, output_size

    const at::Tensor dummy = module.Get(inputs[0]).toTensor();
    const auto output_size = module.Get(inputs[1]).toIntList();

    // 支持 2d(NCHW) / 3d(NCDHW)
    T_CHECK(dummy.ndimension() == 4 && output_size.size() == 2 ||
            dummy.ndimension() == 5 && output_size.size() == 3);

    // 计算 kernel size & stride
    std::vector<int> kernel_size, strides;
    for (int64_t i = 2; i < dummy.ndimension(); ++i) {
      const auto stride = dummy.size(i) / output_size[i - 2];
      strides.push_back(stride);
      kernel_size.push_back(dummy.size(i) - (output_size[i - 2] - 1) * stride);
    }

    layer_desc->windowSize = TrtUtils::ToDims(kernel_size);
    layer_desc->stride = TrtUtils::ToDims(strides);
    layer_desc->padding = nvinfer1::Dims({0});

    if (kind == c10::aten::adaptive_avg_pool2d || kind == c10::aten::adaptive_avg_pool3d) {
      layer_desc->poolingType = nvinfer1::PoolingType::kAVERAGE;
      // layer_desc->averageCountExcludesPadding =
      // !module.Get(inputs[5]).toBool();
    } else if (kind == c10::aten::adaptive_max_pool2d || kind == c10::aten::adaptive_max_pool3d) {
      layer_desc->poolingType = nvinfer1::PoolingType::kMAX;
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
