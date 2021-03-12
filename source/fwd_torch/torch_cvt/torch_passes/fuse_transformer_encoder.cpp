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
            %attention_div, %drop_prop, %gelu_add, %gelu_mul, %gelu_div,
            %none, %zero, %one, %two, %three,
            %minus_one, %minus_two, %bool_f, %bool_t, %epsilon):
        %query : Tensor = aten::linear(%input, %q_kernel, %q_bias)
        %key : Tensor = aten::linear(%input, %k_kernel, %k_bias)
        %value : Tensor = aten::linear(%input, %v_kernel, %v_bias)
        %2311 : int = aten::size(%query, %zero) 
        %2312 : Tensor = prim::NumToTensor(%2311) 
        %2314 : int = aten::size(%query, %one) 
        %2315 : Tensor = prim::NumToTensor(%2314) 
        %2318 : int = aten::Int(%2312) 
        %2320 : int = aten::Int(%2315) 
        %2321 : int[] = prim::ListConstruct(%2318, %2320, %num_heads, %head_size)
        %x65.1 : Tensor = aten::view(%query, %2321) 
        %2324 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %query_layer : Tensor = aten::permute(%x65.1, %2324) 
        %2327 : int = aten::size(%key, %zero) 
        %2328 : Tensor = prim::NumToTensor(%2327) 
        %2330 : int = aten::size(%key, %one) 
        %2331 : Tensor = prim::NumToTensor(%2330) 
        %2334 : int = aten::Int(%2328) 
        %2336 : int = aten::Int(%2331) 
        %2337 : int[] = prim::ListConstruct(%2334, %2336, %num_heads, %head_size)
        %x66.1 : Tensor = aten::view(%key, %2337) 
        %2340 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %key_layer : Tensor = aten::permute(%x66.1, %2340) 
        %2343 : int = aten::size(%value, %zero) 
        %2344 : Tensor = prim::NumToTensor(%2343) 
        %2346 : int = aten::size(%value, %one) 
        %2347 : Tensor = prim::NumToTensor(%2346) 
        %2350 : int = aten::Int(%2344) 
        %2352 : int = aten::Int(%2347) 
        %2353 : int[] = prim::ListConstruct(%2350, %2352, %num_heads, %head_size)
        %x67.1 : Tensor = aten::view(%value, %2353) 
        %2356 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %value_layer : Tensor = aten::permute(%x67.1, %2356) 
        %2360 : Tensor = aten::transpose(%key_layer, %minus_one, %minus_two) 
        %attention_scores17.1 : Tensor = aten::matmul(%query_layer, %2360) 
        %attention_scores18.1 : Tensor = aten::div(%attention_scores17.1, %attention_div) 
        %input85.1 : Tensor = aten::add(%attention_scores18.1, %attention_mask, %one) 
        %input86.1 : Tensor = aten::softmax(%input85.1, %minus_one, %none) 
        %attention_probs : Tensor = aten::dropout(%input86.1, %drop_prop, %bool_f) 
        %context_layer : Tensor = aten::matmul(%attention_probs, %value_layer) 
        %2378 : int[] = prim::ListConstruct(%zero, %two, %one, %three)
        %2379 : Tensor = aten::permute(%context_layer, %2378) 
        %context_layer_1 : Tensor = aten::contiguous(%2379, %zero)
        %2383 : int = aten::size(%context_layer_1, %zero) 
        %2384 : Tensor = prim::NumToTensor(%2383) 
        %2386 : int = aten::size(%context_layer_1, %one) 
        %2387 : Tensor = prim::NumToTensor(%2386) 
        %2390 : int = aten::Int(%2384) 
        %2392 : int = aten::Int(%2387) 
        %2393 : int[] = prim::ListConstruct(%2390, %2392, %hidden_size)
        %input87.1 : Tensor = aten::view(%context_layer_1, %2393) 
        %2870 : Tensor = aten::linear(%input87.1,
                            %attention_output_kernel, %attention_output_bias)
        %hidden_states17.1 : Tensor = aten::dropout(%2870, %drop_prop, %bool_f) 
        %input89.1 : Tensor = aten::add(%hidden_states17.1, %input, %one) 
        %2409 : int[] = prim::ListConstruct(%hidden_size)
        %input_tensor8.1 : Tensor = aten::layer_norm(
                            %input89.1, %2409,
                            %layer_norm_gamma, %layer_norm_beta,
                            %epsilon, %bool_t) 
        %2871 : Tensor = aten::linear(%input_tensor8.1,
                            %intermediate_kernel, %intermediate_bias)
        %2423 : Tensor = aten::div(%2871, %gelu_div) 
        %2424 : Tensor = aten::erf(%2423) 
        %2427 : Tensor = aten::mul(%2871, %gelu_mul) 
        %2430 : Tensor = aten::add(%2424, %gelu_add, %one) 
        %input90.1 : Tensor = aten::mul(%2427, %2430) 
        %2872 : Tensor = aten::linear(%input90.1, %output_kernel, %output_bias)
        %hidden_states18.1 : Tensor = aten::dropout(%2872, %drop_prop, %bool_f) 
        %input92.1 : Tensor = aten::add(%hidden_states18.1, %input_tensor8.1, %one) 
        %2448 : int[] = prim::ListConstruct(%hidden_size)
        %res : Tensor = aten::layer_norm(%input92.1, %2448,
                            %out_layer_norm_gamma, %out_layer_norm_beta,
                            %epsilon, %bool_t) 
        return (%res))IR";

  const std::string dest_pattern = R"IR(
    graphgraph(%input, %q_kernel, %q_bias, %k_kernel, %k_bias, %v_kernel, %v_bias,
            %attention_output_kernel, %attention_output_bias,
            %layer_norm_gamma, %layer_norm_beta,
            %intermediate_kernel, %intermediate_bias,
            %output_kernel, %output_bias,
            %out_layer_norm_gamma, %out_layer_norm_beta,
            %attention_mask, %num_heads, %head_size, %hidden_size,
            %attention_div, %drop_prop, %gelu_add, %gelu_mul, %gelu_div,
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
