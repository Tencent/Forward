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
 * \brief Split 将 Tensor 按照某个维度进行切分
 *
 * Tensor[] = c10::aten::split(%input, %sections, %dim)
 * Tensor[] = c10::aten::split_with_sizes(%input, %sizes, %dim)
 */
template <>
class TLayerDescCreator<TrtSplitDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::split || node->kind() == c10::aten::split_with_sizes ||
           node->kind() == c10::aten::chunk;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtSplitDesc::Create";

    auto layer_desc = std::make_shared<TrtSplitDesc>();

    const auto& inputs = node->inputs();

    layer_desc->splitSize.clear();
    layer_desc->dim = module.Get(inputs[2]).toInt();
    const auto input_dims = DimsOf(module.Get(inputs[0]).toTensor());

    if (node->kind() == c10::aten::split_with_sizes) {
      const auto split_size = module.Get(inputs[1]).toIntList();
      layer_desc->splitSize.assign(split_size.begin(), split_size.end());
    } else if (node->kind() == c10::aten::split || node->kind() == c10::aten::chunk) {
      // 等间距进行切分，如果最后剩余不足，则不足部分作为最后一块的 size
      auto total_size = input_dims.d[layer_desc->dim];

      int chunk_size;

      if (node->kind() == c10::aten::split) {
        chunk_size = module.Get(inputs[1]).toInt();
      } else {
        const int num_chunk = module.Get(inputs[1]).toInt();
        chunk_size = (total_size + num_chunk - 1) / num_chunk;
      }
      for (; total_size > 0; total_size -= chunk_size) {
        layer_desc->splitSize.push_back(std::min(total_size, chunk_size));
      }
    } else {
      LOG(ERROR) << "Unsupported split node type " << node->kind().toQualString();
      return nullptr;
    }

    input_values.push_back(inputs[0]);

    for (auto& size : layer_desc->splitSize) {
      auto chunk_dims = input_dims;
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

FWD_TORCH_NAMESPACE_END
