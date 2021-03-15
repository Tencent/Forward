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
 * \brief Noop 无操作 层描述创建器
 */
template <>
class TLayerDescCreator<TrtNoopDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    // LOG(INFO) << "TorchNoopDescCreator::Check";

    const auto kind = node->kind();

    return kind == c10::prim::ListConstruct || kind == c10::prim::TupleConstruct;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtNoopDesc::Create";

    const auto kind = node->kind();

    LOG(INFO) << kind.toQualString() << " as noop";

    auto layer_desc = std::make_shared<TrtNoopDesc>();

    layer_desc->debug_string = kind.toQualString();

    const auto inputs = node->inputs();

    for (auto input : inputs) {
      Utils::ExtractAllInputs(input, input_values);
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
