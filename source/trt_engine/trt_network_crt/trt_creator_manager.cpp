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
//          Zhaoyi LUO (luozy63@gmail.com)

#include "trt_engine/trt_network_crt/trt_creator_manager.h"

#include "trt_engine/trt_network_crt/layer_creators/trt_activation_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_adaptive_lin_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_adaptive_pool_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_bert_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_cast_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_clamp_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_concatenation_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_constant_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_constant_pad_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_convolution_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_deconvolution_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_element_wise_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_embedding_bag_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_fully_connected_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_gather_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_gelu_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_grid_sampler_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_identity_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_index_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_input_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_lrn_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_mat_mul_add_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_matrix_multiply_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_norm_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_normalization_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_output_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_parametric_relu_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_pooling_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_reduce_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_reflection_pad_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_repeat_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_resize_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_rnnv2_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_scale_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_select_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_separable_conv_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_shuffle_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_slice_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_softmax_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_split_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_topk_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_unary_creator.h"
#include "trt_engine/trt_network_crt/layer_creators/trt_upsample_bilinear_creator.h"

#ifdef ENABLE_TORCH_PLUGIN
#include "trt_engine/trt_network_crt/layer_creators/trt_torch_module_creator.h"
#endif  //  ENABLE_TORCH_PLUGIN

FWD_TRT_NAMESPACE_BEGIN

// register common plugins
REGISTER_TENSORRT_PLUGIN(AdaptiveLINPluginCreator);
REGISTER_TENSORRT_PLUGIN(AdaptivePoolingPluginCreator);
REGISTER_TENSORRT_PLUGIN(CastPluginCreator);
REGISTER_TENSORRT_PLUGIN(ConstantPadPluginCreator);
REGISTER_TENSORRT_PLUGIN(EmbeddingBagPluginCreator);
REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);
REGISTER_TENSORRT_PLUGIN(IndexPluginCreator);
REGISTER_TENSORRT_PLUGIN(LayerNormPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(NormalizationPluginCreator);
REGISTER_TENSORRT_PLUGIN(NormPluginCreator);
REGISTER_TENSORRT_PLUGIN(ReducePluginCreator);
REGISTER_TENSORRT_PLUGIN(ReflectionPadding2DPluginCreator);
REGISTER_TENSORRT_PLUGIN(SplitPluginCreator);
REGISTER_TENSORRT_PLUGIN(UpsampleBilinear2DPluginCreator);

#ifdef ENABLE_TORCH_PLUGIN
REGISTER_TENSORRT_PLUGIN(TorchModulePluginCreator);
#endif  //  ENABLE_TORCH_PLUGIN

LayerCreatorManager::LayerCreatorManager() {
  // 这里注册时候，按照字母顺序吧，便于查找

  using namespace trt_;

  initLibNvInferPlugins(getLogger(), "");

  RegisterCreator<TrtActivationDesc>();
  RegisterCreator<TrtAdaptiveLinDesc>();
  RegisterCreator<TrtAdaptivePoolDesc>();
  RegisterCreator<TrtBertDesc>();
  RegisterCreator<TrtCastDesc>();
  RegisterCreator<TrtClampDesc>();
  RegisterCreator<TrtConcatenationDesc>();
  RegisterCreator<TrtConstantDesc>();
  RegisterCreator<TrtConstantPadDesc>();
  RegisterCreator<TrtConvolutionDesc>();
  RegisterCreator<TrtDeconvolutionDesc>();
  RegisterCreator<TrtElementWiseDesc>();
  RegisterCreator<TrtEmbeddingBagDesc>();
  RegisterCreator<TrtFullyConnectedDesc>();
  RegisterCreator<TrtGatherDesc>();
  RegisterCreator<TrtGeluDesc>();
  RegisterCreator<TrtGridSamplerDesc>();
  RegisterCreator<TrtIdentityDesc>();
  RegisterCreator<TrtIndexDesc>();
  RegisterCreator<TrtInputDesc>();
  RegisterCreator<TrtNormalizationDesc>();
  RegisterCreator<TrtLRNDesc>();
  RegisterCreator<TrtMatrixMultiplyDesc>();
  // RegisterCreator<TrtNoopDesc>();
  RegisterCreator<TrtNormDesc>();
  RegisterCreator<TrtOutputDesc>();
  RegisterCreator<TrtParametricReLUDesc>();
  RegisterCreator<TrtPoolingDesc>();
  RegisterCreator<TrtReflectionPadDesc>();
  RegisterCreator<TrtReduceDesc>();
  RegisterCreator<TrtRepeatDesc>();
  RegisterCreator<TrtResizeDesc>();
  RegisterCreator<TrtRNNv2Desc>();
  RegisterCreator<TrtScaleDesc>();
  RegisterCreator<TrtSelectDesc>();
  RegisterCreator<TrtSeparableConvDesc>();
  RegisterCreator<TrtShuffleDesc>();
  RegisterCreator<TrtSliceDesc>();
  RegisterCreator<TrtSplitDesc>();
  RegisterCreator<TrtSoftmaxDesc>();
  RegisterCreator<TrtTopKDesc>();
  RegisterCreator<TrtUnaryDesc>();
  RegisterCreator<TrtUpsampleBilinearDesc>();

  RegisterCreator<TrtMatMulAddDesc>();

#ifdef ENABLE_TORCH_PLUGIN
  RegisterCreator<TrtTorchModuleDesc>();
#endif  // ENABLE_TORCH_PLUGIN
}

ILayerCreator* LayerCreatorManager::FindCreator(const std::string& layer_name) const {
  auto iter = name_to_creator_mapping_.find(layer_name);
  if (iter == name_to_creator_mapping_.end()) {
    LOG(ERROR) << "Could not find layer creator << " << layer_name;
    return nullptr;
  }

  return iter->second.get();
}

FWD_TRT_NAMESPACE_END
