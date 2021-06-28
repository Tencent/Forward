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

#include "fwd_torch/torch_cvt/torch_desc_manager.h"

#include <memory>
#include <string>
#include <vector>

#include "common/fwd_common.h"
#include "fwd_torch/torch_cvt/torch_helper.h"
#include "fwd_torch/torch_cvt/torch_module.h"
#include "torch_desc_creators/torch_activation_creator.h"
#include "torch_desc_creators/torch_adaptive_lin_creator.h"
#include "torch_desc_creators/torch_adaptive_pool_creator.h"
#include "torch_desc_creators/torch_bert_creator.h"
#include "torch_desc_creators/torch_cast_creator.h"
#include "torch_desc_creators/torch_clamp_creator.h"
#include "torch_desc_creators/torch_concatenation_creator.h"
#include "torch_desc_creators/torch_constant_creator.h"
#include "torch_desc_creators/torch_constant_pad_creator.h"
#include "torch_desc_creators/torch_convolution_creator.h"
#include "torch_desc_creators/torch_element_wise_creator.h"
#include "torch_desc_creators/torch_embedding_bag_creator.h"
#include "torch_desc_creators/torch_gather_creator.h"
#include "torch_desc_creators/torch_gelu_creator.h"
#include "torch_desc_creators/torch_grid_sampler_creator.h"
#include "torch_desc_creators/torch_identity_creator.h"
#include "torch_desc_creators/torch_index_creator.h"
#include "torch_desc_creators/torch_linear_creator.h"
#include "torch_desc_creators/torch_lrn_creator.h"
#include "torch_desc_creators/torch_matrix_multiply_creator.h"
#include "torch_desc_creators/torch_noop_creator.h"
#include "torch_desc_creators/torch_norm_creator.h"
#include "torch_desc_creators/torch_normalization_creator.h"
#include "torch_desc_creators/torch_pooling_creator.h"
#include "torch_desc_creators/torch_prelu_creator.h"
#include "torch_desc_creators/torch_reduce_creator.h"
#include "torch_desc_creators/torch_reflection_pad_creator.h"
#include "torch_desc_creators/torch_repeat_creator.h"
#include "torch_desc_creators/torch_resize_creator.h"
#include "torch_desc_creators/torch_rnn_creator.h"
#include "torch_desc_creators/torch_shuffle_creator.h"
#include "torch_desc_creators/torch_slice_creator.h"
#include "torch_desc_creators/torch_softmax_creator.h"
#include "torch_desc_creators/torch_split_creator.h"
#include "torch_desc_creators/torch_unary_creator.h"
#include "torch_desc_creators/torch_upsample_bilinear_creator.h"

#ifdef ENABLE_TORCH_PLUGIN
#include "torch_desc_creators/torch_submodule_creator.h"
#endif

FWD_TORCH_NAMESPACE_BEGIN

TorchDescManager::TorchDescManager() {
  RegisterCreator<TrtBertDesc>();
  RegisterCreator<TrtAdaptiveLinDesc>();

  layer_desc_creators_.push_back(std::make_shared<TorchNormalizationCreator>());
  layer_desc_creators_.push_back(std::make_shared<TorchConvolutionCreator>());
  layer_desc_creators_.push_back(std::make_shared<TorchAddmmCreator>());

  // 这里注册的时候，将复杂的描述放在上面，会先检测复杂的模式
  RegisterCreator<TrtLRNDesc>();
  RegisterCreator<TrtNormDesc>();  // Norm: norm->div > ElementWise
  RegisterCreator<TrtConstantPadDesc>();
  RegisterCreator<TrtShuffleDesc>();
  RegisterCreator<TrtActivationDesc>();
  RegisterCreator<TrtEmbeddingBagDesc>();
  RegisterCreator<TrtGatherDesc>();
  RegisterCreator<TrtRNNv2Desc>();

  RegisterCreator<TrtCastDesc>();
  RegisterCreator<TrtConstantDesc>();
  RegisterCreator<TrtSplitDesc>();
  RegisterCreator<TrtClampDesc>();
  RegisterCreator<TrtConcatenationDesc>();
  RegisterCreator<TrtGatherDesc>();
  RegisterCreator<TrtGeluDesc>();
  RegisterCreator<TrtGridSamplerDesc>();
  RegisterCreator<TrtIndexDesc>();
  RegisterCreator<TrtPoolingDesc>();
  RegisterCreator<TrtReflectionPadDesc>();
  RegisterCreator<TrtMatrixMultiplyDesc>();
  RegisterCreator<TrtUpsampleBilinearDesc>();
  RegisterCreator<TrtResizeDesc>();
  RegisterCreator<TrtSliceDesc>();
  RegisterCreator<TrtSoftmaxDesc>();
  RegisterCreator<TrtUnaryDesc>();
  RegisterCreator<TrtReduceDesc>();
  RegisterCreator<TrtParametricReLUDesc>();
  RegisterCreator<TrtAdaptivePoolDesc>();
  RegisterCreator<TrtElementWiseDesc>();
  RegisterCreator<TrtRepeatDesc>();

  // 这种简单的模式就放在下面
  RegisterCreator<TrtIdentityDesc>();
  RegisterCreator<TrtNoopDesc>();

#ifdef ENABLE_TORCH_PLUGIN
  RegisterCreator<TrtTorchModuleDesc>();
#endif
}

ILayerDescCreator* TorchDescManager::FindDescCreator(const JitNode* node,
                                                     const TorchModule& module) {
  // 查找对应的层描述创建器
  for (auto creator : layer_desc_creators_) {
    if (creator->Check(node, module)) {
      return creator.get();
    }
  }

  LOG(ERROR) << "Could not find layer create for node " << node->kind().toQualString()
             << ": output " << node->outputs()[0]->debugName();

  return nullptr;
}

FWD_TORCH_NAMESPACE_END
