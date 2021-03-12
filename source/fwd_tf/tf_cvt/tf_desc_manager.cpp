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

#include "fwd_tf/tf_cvt/tf_desc_manager.h"

#include "fwd_tf/tf_cvt/tf_desc_creators/tf_activation_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_batch_norm_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_bert_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_clamp_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_concatenation_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_convolution_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_dense_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_element_wise_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_embedding_bag_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_gather_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_identity_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_matmul_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_noop_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_pad_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_pooling_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_reduce_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_select_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_shuffle_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_slice_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_softmax_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/tf_split_creator.h"

FWD_TF_NAMESPACE_BEGIN

TfDescManager::TfDescManager() {
  // 这里注册的时候，将复杂的描述放在上面，会先检测复杂的模式0
  RegisterCreator<TrtNormalizationDesc>();
  RegisterCreator<TrtBertDesc>();
  RegisterCreator<TrtFullyConnectedDesc>();
  RegisterCreator<TrtConvolutionDesc>();
  RegisterCreator<TrtMatrixMultiplyDesc>();
  RegisterCreator<TrtActivationDesc>();
  RegisterCreator<TrtPoolingDesc>();
  RegisterCreator<TrtEmbeddingBagDesc>();
  RegisterCreator<TrtIdentityDesc>();
  // Identity>Select(判断mask pedding)
  RegisterCreator<TrtSelectDesc>();
  // Select优先级高于ElementWise
  RegisterCreator<TrtElementWiseDesc>();
  RegisterCreator<TrtReduceDesc>();
  RegisterCreator<TrtSliceDesc>();
  RegisterCreator<TrtSplitDesc>();
  RegisterCreator<TrtSoftmaxDesc>();
  RegisterCreator<TrtClampDesc>();
  RegisterCreator<TrtGatherDesc>();
  RegisterCreator<TrtConstantPadDesc>();
  RegisterCreator<TrtConcatenationDesc>();
  RegisterCreator<TrtShuffleDesc>();

  // 这种简单的模式就放在下面
  RegisterCreator<TrtNoopDesc>();
}

ILayerDescCreator* TfDescManager::FindDescCreator(const Operation& op) {
  // 查找对应的层描述创建器
  for (const auto& creator : layer_desc_creators_) {
    if (creator->Check(op)) {
      return creator.get();
    }
  }

  LOG(ERROR) << "Could not find layer create for operation " << op.Name() << "[" << op.OpType()
             << "]";

  return nullptr;
}

FWD_TF_NAMESPACE_END
