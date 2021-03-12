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

FWD_TORCH_NAMESPACE_BEGIN
/**
 * \brief AdaptivePooling 层描述创建器
 */
template <>
class TLayerDescCreator<TrtAdaptivePoolDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    // LOG(INFO) << "TrtAdaptivePoolDesc::Check";

    const auto kind = node->kind();

    // TODO(Ao Li): 增加对 1d 的支持

    return kind == c10::aten::adaptive_avg_pool2d || kind == c10::aten::adaptive_avg_pool3d ||
           kind == c10::aten::adaptive_max_pool2d || kind == c10::aten::adaptive_max_pool3d;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtAdaptivePoolDesc::Create";

    auto layer_desc = std::make_shared<TrtAdaptivePoolDesc>();

    const auto kind = node->kind();

    const auto inputs = node->inputs();

    input_values.push_back(inputs[0]);

    // Input 0, kind = input
    // Input 1, kind = c10::prim::ListConstruct, output_size

    const auto output_size = module.Get(inputs[1]).toIntList();
    T_CHECK(output_size.size() == 2 || output_size.size() == 3);
    layer_desc->output_size.assign(output_size.begin(), output_size.end());

    if (kind == c10::aten::adaptive_avg_pool2d || kind == c10::aten::adaptive_avg_pool3d) {
      layer_desc->pooling_type = nvinfer1::PoolingType::kAVERAGE;
    } else if (kind == c10::aten::adaptive_max_pool2d || kind == c10::aten::adaptive_max_pool3d) {
      layer_desc->pooling_type = nvinfer1::PoolingType::kMAX;
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
