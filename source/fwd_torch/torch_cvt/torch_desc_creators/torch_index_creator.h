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
 * \brief Index 层描述创建器
 */
template <>
class TLayerDescCreator<TrtIndexDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::index;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtIndexDesc::Create";

    auto layer_desc = std::make_shared<TrtIndexDesc>();

    const auto kind = node->kind();

    // Input 0, kind = input
    // Input 1, kind = TensorList/GenericList: indices
    const auto inputs = node->inputs();

    input_values.push_back(inputs[0]);

    layer_desc->axis.resize(8, 0);
    layer_desc->nbIndexDims = 0;
#if FWD_TORCH_VERSION > 160
    if (module.Get(inputs[1]).isList()) {
      const auto indices = module.Get(inputs[1]).toList();
#else
    if (module.Get(inputs[1]).isGenericList()) {
      const auto indices = module.Get(inputs[1]).toGenericList();
#endif
      layer_desc->nbDims = indices.size();

      for (size_t i = 0; i < indices.size(); i++) {
        if (indices.get(i).isTensor()) {
          const auto index_dim_i = indices.get(i).toTensor().toType(c10::ScalarType::Int);
          layer_desc->axis[i] = 1;
          layer_desc->nbIndexDims++;
          layer_desc->nbIndex = index_dim_i.numel();

          int* index_ptr = static_cast<int*>(index_dim_i.data_ptr());
          for (int p = 0; p < index_dim_i.numel(); p++) {
            layer_desc->indices.push_back(index_ptr[p]);
          }
        }
      }
    } else if (module.Get(inputs[1]).isTensorList()) {
      const auto indices = module.Get(inputs[1]).toTensorList();
      layer_desc->nbDims = indices.size();

      for (size_t i = 0; i < indices.size(); i++) {
        const auto index_dim_i = indices.get(i).toType(c10::ScalarType::Int);
        layer_desc->axis[i] = 1;
        layer_desc->nbIndexDims++;
        layer_desc->nbIndex = index_dim_i.numel();

        int* index_ptr = static_cast<int*>(index_dim_i.data_ptr());
        for (int p = 0; p < index_dim_i.numel(); p++) {
          layer_desc->indices.push_back(index_ptr[p]);
        }
      }
    }

    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
