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

// NOTE: TensorRT 7.0 及其以下版本对 Resize 支持的不是那么完善
// 这个 plugin 针对 7.0 及其以下版本的 TensorRT 实现 Upsample Bilinear 2D

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Upsample Bilinear 2D 层描述创建器
 */
template <>
class TLayerDescCreator<TrtUpsampleBilinearDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    // 仅仅在 TensorRT 版本低于 7.1.3 时才启用 plugin
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH < 7103
    return node->kind() == c10::aten::upsample_bilinear2d;
#else
    return false;
#endif
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtUpsampleBilinearDesc::Create";

    const auto inputs = node->inputs();

    // c10::aten::upsample_bilinear2d(Tensor self, int[] output_size, bool
    // align_corners, float scales_1, float scales_2) -> Tensor

    input_values.push_back(inputs[0]);

    const at::Tensor dummy = module.Get(inputs[0]).toTensor();
    T_CHECK_EQ(dummy.ndimension(), 4);  // TODO(Ao Li): 暂时只支持 upsample 2d

    auto layer_desc = std::make_shared<TrtUpsampleBilinearDesc>();

    // torch1.7.0 模型中出现第二个参数为None时，后面的 scale 参数会生效
    if (module.Get(inputs[1]).isNone()) {
      c10::List<double> scales;
      if (node->kind() == c10::aten::upsample_bilinear2d) {
        layer_desc->alignCorners = module.Get(inputs[2]).toBool();
        scales = module.Get(inputs[3]).toDoubleList();
      } else {
        scales = module.Get(inputs[2]).toDoubleList();
      }
      T_CHECK_EQ(scales.size(), 2);
      layer_desc->scale_h = scales[0];
      layer_desc->scale_w = scales[1];
      layer_desc->output_h = dummy.size(2) * layer_desc->scale_h;
      layer_desc->output_w = dummy.size(3) * layer_desc->scale_w;
    } else {
      const auto output_size = module.Get(inputs[1]).toIntList();
      layer_desc->output_h = output_size[0];
      layer_desc->output_w = output_size[1];
      layer_desc->alignCorners = module.Get(inputs[2]).toBool();
      if (inputs.size() == 5) {
        layer_desc->scale_h = module.Get(inputs[3]).toDouble();
        layer_desc->scale_w = module.Get(inputs[4]).toDouble();
      }
    }
    return layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
