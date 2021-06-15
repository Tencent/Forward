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

#include "fwd_torch/torch_cvt/torch_passes/fuse_transformer_encoder.h"

#include <easylogging++.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <common/common_macros.h>
#include <string>

static auto bert_registry = torch::RegisterOperators(
    // 注意：此处是直接注册到 torch 的库中，属于进程全局注册，需避免重复注册
    "fwd::transformer_encoder",
    [](at::Tensor input, at::Tensor q_kernel, at::Tensor q_bias, at::Tensor k_kernel,
       at::Tensor k_bias, at::Tensor v_kernel, at::Tensor v_bias,
       at::Tensor attention_output_kernel, at::Tensor attention_output_bias,
       at::Tensor layer_norm_gamma, at::Tensor layer_norm_beta, at::Tensor intermediate_kernel,
       at::Tensor intermediate_bias, at::Tensor output_kernel, at::Tensor output_bias,
       at::Tensor out_layer_norm_gamma, at::Tensor out_layer_norm_beta, at::Tensor attention_mask,
       int64_t num_heads, int64_t head_size, int64_t hidden_size) {
      // TODO(Ao Li): 此处不必实现完整操作，仅为方便 transformer 层识别，
      // 但需保证输出 Tensor shape正确，因为 evalAll 时需要该 Tensor
      // shape
      return input;
    });

void torch::pass::FuseTransformerEncoder(std::shared_ptr<torch::jit::Graph>& graph) {
  // 注册 custom op, 仅为了 forward 能够 parse 成功，并不提供 torch 计算
  if (jit::getAllOperatorsFor(Symbol::fromQualString("fwd::transformer_encoder")).empty()) {
    LOG(WARNING) << "fwd::transformer_encoder has not been registered in torch.";
  }

  const std::string src_pattern = R"IR(
    graph(%input, %q_kernel, %q_bias, %k_kernel, %k_bias, %v_kernel, %v_bias,
            %attention_output_kernel, %attention_output_bias,
            %layer_norm_gamma, %layer_norm_beta,
            %intermediate_kernel, %intermediate_bias,
            %output_kernel, %output_bias,
            %out_layer_norm_gamma, %out_layer_norm_beta,
            %attention_mask, %num_heads, %head_size, %hidden_size,
            %attention_div, %drop_prop,
            %none, %zero, %one, %two, %three,
            %minus_one, %minus_two, %bool_f, %bool_t, %epsilon):
        
        %1394 : Tensor = aten::linear(%input, %q_kernel, %q_bias)
        %1395 : Tensor = aten::linear(%input, %k_kernel, %k_bias)
        %1094 : int = aten::size(%1395, %zero) 
        %1095 : int = aten::size(%1395, %one) 
        %1096 : int[] = prim::ListConstruct(%1094, %1095, %num_heads, %head_size)
        %x.1 : Tensor = aten::view(%1395, %1096) 
        %1098 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %key_layer.1 : Tensor = aten::permute(%x.1, %1098) 
        %1396 : Tensor = aten::linear(%input, %v_kernel, %v_bias)
        %1105 : int = aten::size(%1396, %zero) 
        %1106 : int = aten::size(%1396, %one) 
        %1107 : int[] = prim::ListConstruct(%1105, %1106, %num_heads, %head_size)
        %x0.1 : Tensor = aten::view(%1396, %1107) 
        %1109 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %value_layer.1 : Tensor = aten::permute(%x0.1, %1109) 
        %1111 : int = aten::size(%1394, %zero) 
        %1112 : int = aten::size(%1394, %one) 
        %1113 : int[] = prim::ListConstruct(%1111, %1112, %num_heads, %head_size)
        %x1.1 : Tensor = aten::view(%1394, %1113) 
        %1115 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %query_layer.1 : Tensor = aten::permute(%x1.1, %1115) 
        %1117 : Tensor = aten::transpose(%key_layer.1, %minus_one, %minus_two) 
        %attention_scores.1 : Tensor = aten::matmul(%query_layer.1, %1117) 
        %attention_scores0.1 : Tensor = aten::div(%attention_scores.1, %attention_div) 
        %input.2 : Tensor = aten::add(%attention_scores0.1, %attention_mask, %one) 
        %input0.25 : Tensor = aten::softmax(%input.2, %minus_one, %none) 
        %attention_probs.1 : Tensor = aten::dropout(%input0.25, %drop_prop, %bool_f) 
        %context_layer.1 : Tensor = aten::matmul(%attention_probs.1, %value_layer.1) 
        %1124 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %1125 : Tensor = aten::permute(%context_layer.1, %1124) 
        %context_layer0.1 : Tensor = aten::contiguous(%1125, %zero) 
        %1127 : int = aten::size(%context_layer0.1, %zero) 
        %1128 : int = aten::size(%context_layer0.1, %one) 
        %1129 : int[] = prim::ListConstruct(%1127, %1128, %hidden_size)
        %input1.1 : Tensor = aten::view(%context_layer0.1, %1129) 
        %1397 : Tensor = aten::linear(%input1.1, %attention_output_kernel, %attention_output_bias)
        %hidden_states.3 : Tensor = aten::dropout(%1397, %drop_prop, %bool_f) 
        %input.3 : Tensor = aten::add(%hidden_states.3, %input, %one) 
        %1142 : int[] = prim::ListConstruct(%hidden_size)
        %input_tensor.1 : Tensor = aten::layer_norm(%input.3, %1142, %layer_norm_gamma, %layer_norm_beta, %epsilon, %bool_t) 
        %1398 : Tensor = aten::linear(%input_tensor.1, %intermediate_kernel, %intermediate_bias)
        %input.4 : Tensor = aten::gelu(%1398) 
        %1399 : Tensor = aten::linear(%input.4, %output_kernel, %output_bias)
        %hidden_states.2 : Tensor = aten::dropout(%1399, %drop_prop, %bool_f) 
        %input.51 : Tensor = aten::add(%hidden_states.2, %input_tensor.1, %one) 
        %1162 : int[] = prim::ListConstruct(%hidden_size)
        %res : Tensor = aten::layer_norm(%input.51, %1162, %out_layer_norm_gamma, %out_layer_norm_beta, %epsilon, %bool_t) 
        return (%res))IR";
  const std::string dest_pattern = R"IR(
    graphgraph(%input, %q_kernel, %q_bias, %k_kernel, %k_bias, %v_kernel, %v_bias,
            %attention_output_kernel, %attention_output_bias,
            %layer_norm_gamma, %layer_norm_beta,
            %intermediate_kernel, %intermediate_bias,
            %output_kernel, %output_bias,
            %out_layer_norm_gamma, %out_layer_norm_beta,
            %attention_mask, %num_heads, %head_size, %hidden_size,
            %attention_div, %drop_prop, 
            %none, %zero, %one, %two, %three,
            %minus_one, %minus_two, %bool_f, %bool_t, %epsilon):
        %res = fwd::transformer_encoder(%input, %q_kernel, %q_bias,
                  %k_kernel, %k_bias, %v_kernel, %v_bias,
                  %attention_output_kernel, %attention_output_bias,
                  %layer_norm_gamma, %layer_norm_beta,
                  %intermediate_kernel, %intermediate_bias,
                  %output_kernel, %output_bias,
                  %out_layer_norm_gamma, %out_layer_norm_beta,
                  %attention_mask, %num_heads, %head_size, %hidden_size)
        return (%res))IR";

  jit::SubgraphRewriter transformer_encoder_fuser;
  transformer_encoder_fuser.RegisterRewritePattern(src_pattern, dest_pattern);
  transformer_encoder_fuser.runOnGraph(graph);
}
