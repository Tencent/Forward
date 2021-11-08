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
 * \brief Resize 层描述创建器
 */
template <>
class TLayerDescCreator<TrtResizeDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind() == c10::aten::upsample_bilinear2d ||
           node->kind() == c10::aten::upsample_nearest2d ||
           node->kind() == c10::aten::upsample_trilinear3d ||
           node->kind() == c10::aten::upsample_nearest3d;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtResizeDesc::Create";

    const auto inputs = node->inputs();

    // c10::aten::upsample_bilinear2d(Tensor self, int[] output_size, bool
    // align_corners, float scales_1, float scales_2) -> Tensor
    // c10::aten::upsample_nearest2d(Tensor self, int[] output_size, float
    // scales_1, float scales_2) -> Tensor

    input_values.push_back(inputs[0]);

    const at::Tensor dummy = module.Get(inputs[0]).toTensor();
    T_CHECK(dummy.ndimension() == 4 || dummy.ndimension() == 5);  // TODO(Ao Li): 暂时只支持 upsample 2d

    auto layer_desc = std::make_shared<TrtResizeDesc>();
    layer_desc->outputDimensions.nbDims = dummy.ndimension();
    for (int i = 0; i < dummy.ndimension(); ++i) {
      layer_desc->outputDimensions.d[i] = dummy.size(i);
    }

    // torch1.7.0 模型中出现第二个参数为None时，后面的 scale 参数会生效
    if (module.Get(inputs[1]).isNone()) {
      if (!CreateWithScales(node, module, layer_desc.get())) return {};
    } else {
      if (!CreateWithoutScales(node, module, layer_desc.get())) return {};
    }

    return layer_desc;
  }

 private:
  bool CreateWithScales(const JitNode* node, const TorchModule& module, TrtResizeDesc* layer_desc) {
    const auto inputs = node->inputs();
    const at::Tensor dummy = module.Get(inputs[0]).toTensor();

    c10::List<double> scales;
    if (node->kind() == c10::aten::upsample_bilinear2d ||
        node->kind() == c10::aten::upsample_trilinear3d) {
      layer_desc->resizeMode = nvinfer1::ResizeMode::kLINEAR;
      layer_desc->alignCorners = module.Get(inputs[2]).toBool();
      scales = module.Get(inputs[3]).toDoubleList();
    } else {
      layer_desc->resizeMode = nvinfer1::ResizeMode::kNEAREST;
      scales = module.Get(inputs[2]).toDoubleList();
    }

    T_CHECK(scales.size() == 2 || scales.size() == 3);

    // 不使用 scales 是因为在 bilinear2d 结果会有问题
    for (int i = 0; i < scales.size(); ++i) {
      layer_desc->outputDimensions.d[2 + i] = dummy.size(2 + i) * scales[i];
    }

    return true;
  }

  bool CreateWithoutScales(const JitNode* node, const TorchModule& module,
                           TrtResizeDesc* layer_desc) {
    const auto inputs = node->inputs();
    const auto output_size = module.Get(inputs[1]).toIntVector();
    for (int i = 0; i < output_size.size(); ++i) {
      layer_desc->outputDimensions.d[2 + i] = output_size[i];
    }

    if (node->kind() == c10::aten::upsample_bilinear2d ||
        node->kind() == c10::aten::upsample_trilinear3d) {
      layer_desc->resizeMode = nvinfer1::ResizeMode::kLINEAR;
      layer_desc->alignCorners = module.Get(inputs[2]).toBool();
      if (inputs.size() == 5) {  // 认为这里不会出现有 scale 的情况
        T_CHECK(module.Get(inputs[3]).isNone() && module.Get(inputs[4]).isNone());
      }
    } else {
      layer_desc->resizeMode = nvinfer1::ResizeMode::kNEAREST;
      if (inputs.size() == 4) {
        T_CHECK(module.Get(inputs[2]).isNone() && module.Get(inputs[3]).isNone());
      }
    }

    return true;
  }
};

FWD_TORCH_NAMESPACE_END
