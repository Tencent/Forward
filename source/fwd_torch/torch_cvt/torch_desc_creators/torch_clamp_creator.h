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
 * \brief Clamp 层描述创建器
 */
template <>
class TLayerDescCreator<TrtClampDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    // LOG(INFO) << "TrtClampDesc::Check";

    const auto kind = node->kind();

#ifdef NEW_TORCH_API
    if (kind == c10::aten::clamp_) return true;
#endif

    return kind == c10::aten::clamp || kind == c10::aten::hardtanh ||
           kind == c10::Symbol::fromQualString("aten::hardtanh_");
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtClampDesc::Create";

    const auto inputs = node->inputs();

    // Input 0: Tensor input
    // Input 1: c10::optional<Scalar> min
    // Input 2: c10::optional<Scalar> max

    input_values.push_back(inputs[0]);

    auto layer_desc = std::make_shared<TrtClampDesc>();
    if (module.Get(inputs[1]).isNone()) {
      layer_desc->has_min = false;
    } else {
      auto value = module.Get(inputs[1]);
      layer_desc->min = value.isDouble() ? value.toDouble() : value.toInt();
    }

    if (module.Get(inputs[2]).isNone()) {
      layer_desc->has_max = false;
    } else {
      auto value = module.Get(inputs[2]);
      layer_desc->max = value.isDouble() ? value.toDouble() : value.toInt();
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
