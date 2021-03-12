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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Slice 层描述创建器
 */
template <>
class TLayerDescCreator<TrtSliceDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::slice || node->kind() == c10::aten::select;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TorchSliceDesc::Create";

    auto layer_desc = std::make_shared<TrtSliceDesc>();

    const JitValue* output = node->output();

    // TODO(Ao Li): 适配 TRT 维度与输出维度相关，这里暂时通过 Eval 拿取
    const at::Tensor dummy = module.Get(output).toTensor();
    // 初始化
    layer_desc->start.nbDims = dummy.ndimension();
    layer_desc->stride.nbDims = dummy.ndimension();
    layer_desc->size.nbDims = dummy.ndimension();
    for (int i = 0; i < dummy.ndimension(); ++i) {
      layer_desc->start.d[i] = 0;
      layer_desc->stride.d[i] = 1;
      layer_desc->size.d[i] = dummy.size(i);
    }

    const JitNode* prev_node = node;

    while (node->kind() == c10::aten::slice || node->kind() == c10::aten::select) {
      if (node->kind() == c10::aten::select) {
        if (!HandleSelect(node, module, layer_desc)) break;
      } else {
        if (!HandleSlice(node, module, layer_desc, prev_node)) break;
      }

      prev_node = node;
      node = node->inputs()[0]->node();
    }

    // 处理常量 slice
    if (node->kind() == c10::prim::GetAttr || node->kind() == c10::prim::Constant ||
        node->kind() == c10::aten::zeros_like || node->kind() == c10::aten::zeros) {
      auto tensor = module.Get(node->output()).toTensor();
      for (int i = 0; i < layer_desc->start.nbDims; ++i) {
        tensor =
            tensor.slice(i, layer_desc->start.d[i], layer_desc->start.d[i] + layer_desc->size.d[i],
                         layer_desc->stride.d[i]);
      }
      tensor = tensor.contiguous();  // must be contiguous
      auto constant_desc = std::make_shared<TrtConstantDesc>();
      constant_desc->weights = Utils::ToFwdWeights(tensor);
      constant_desc->dimensions = Utils::DimsOf(tensor);
      input_values.push_back(nullptr);  // constant input
      return constant_desc;
    }

    auto start_vec = TrtUtils::ToVector(layer_desc->start);

    for (int i = 0; i < layer_desc->size.nbDims; ++i) {
      // TODO(yzx): 这里默认若是负值，则对应维度的 size 取全集。
      if (layer_desc->size.d[i] < 0) {
        layer_desc->size.d[i] = std::numeric_limits<int>::max();
        layer_desc->dynamic_size = true;
      }

      if (layer_desc->start.d[i] < 0) {
        layer_desc->start.d[i] = std::numeric_limits<int>::max();
        layer_desc->dynamic_start = true;
      }
    }

    if (layer_desc->dynamic_start) {
      layer_desc->weight_map["start_raw"] = FwdWeights(TrtUtils::ToVector(layer_desc->start));
      for (auto& s : start_vec) s = s < 0 ? s : 0;
      layer_desc->weight_map["start_negative"] = FwdWeights(start_vec);
    }
    if (layer_desc->dynamic_size) {
      layer_desc->weight_map["size"] = FwdWeights(TrtUtils::ToVector(layer_desc->size));
    }

    // 非常量 slice
    input_values.push_back(prev_node->inputs()[0]);
    return layer_desc;
  }

 private:
  /**
   * \brief 创建适用于 Select 的层描述
   * \param node
   * \param module
   * \param layer_desc
   * \return
   */
  bool HandleSelect(const JitNode* node, const TorchModule& module,
                    std::shared_ptr<TrtSliceDesc> layer_desc) {
    const auto output_dummy = module.Get(node->output()).toTensor();
    const auto dummy = module.Get(node->inputs()[0]).toTensor();
    // Check input dimension
    if (output_dummy.ndimension() != dummy.ndimension() - 1) {
      return false;
    }

    // Update inputs
    const auto inputs = node->inputs();

    // Input 0: Tensor input
    // Input 1: int dim
    // Input 2: int index
    const auto dim = module.Get(inputs[1]).toInt();
    const auto index = module.Get(inputs[2]).toInt();

    // 升维度
    layer_desc->start.nbDims += 1;
    layer_desc->stride.nbDims += 1;
    layer_desc->size.nbDims += 1;

    for (int i = layer_desc->size.nbDims - 1; i > dim; --i) {
      layer_desc->start.d[i] = layer_desc->start.d[i - 1];
      layer_desc->stride.d[i] = layer_desc->stride.d[i - 1];
      layer_desc->size.d[i] = layer_desc->size.d[i - 1];
    }

    layer_desc->start.d[dim] = index;
    layer_desc->stride.d[dim] = 1;
    layer_desc->size.d[dim] = 1;
    return true;
  }

  /**
   * \brief 创建适用于 Slice 的层描述
   * \param node
   * \param module
   * \param layer_desc
   * \param prev_node
   * \return
   */
  bool HandleSlice(const JitNode* node, const TorchModule& module,
                   std::shared_ptr<TrtSliceDesc> layer_desc, const JitNode* prev_node) {
    const auto prev_dummy = module.Get(prev_node->inputs()[0]).toTensor();
    const auto dummy = module.Get(node->inputs()[0]).toTensor();
    // Check input dimension
    if (prev_dummy.ndimension() != dummy.ndimension()) {
      return false;
    }

    // Update inputs
    const auto inputs = node->inputs();

    // Input 0: Tensor input
    // Input 1: int dim
    // Input 2: int start
    // Input 3: int end
    // Input 4: int step
    const auto dim = module.Get(inputs[1]).toInt();
    const auto start = module.Get(inputs[2]).toInt();
    auto end = module.Get(inputs[3]).toInt();
    const auto step = module.Get(inputs[4]).toInt();

    layer_desc->start.d[dim] = layer_desc->start.d[dim] * step + start;
    layer_desc->stride.d[dim] *= step;
    end = (end == INT64_MAX || end < 0) ? dummy.size(dim) : end;
    int size = (end - start + step - 1) / step;
#ifdef USE_DYNAMIC_BATCH
    if (dim == 0 && size == dummy.size(dim)) size = -1;
#endif
    layer_desc->size.d[dim] = std::min(layer_desc->size.d[dim], size);
    return true;
  }
};

FWD_TORCH_NAMESPACE_END
