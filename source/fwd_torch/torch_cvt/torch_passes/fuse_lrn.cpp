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

#include "fwd_torch/torch_cvt/torch_passes/fuse_lrn.h"

#include <easylogging++.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <string>

static auto lrn_registry = torch::RegisterOperators(
    // 注意：此处是直接注册到 torch 的库中，属于进程全局注册，需避免重复注册
    "fwd::lrn", [](at::Tensor input, int64_t size, at::Tensor alpha, double beta, at::Tensor k,
                   int64_t zero, int64_t one, int64_t two, int64_t three, int64_t minus_one) {
      // TODO(Ao Li): 此处不必实现完整操作，仅为方便 LRN 层识别，
      // 但需保证输出 Tensor shape正确，因为 evalAll 时需要该 Tensor shape
      return input;
    });

void torch::pass::FuseLrn(std::shared_ptr<torch::jit::Graph>& graph) {
  // 注册 custom op, 仅为了 forward 能够 parse 成功，并不提供 torch 计算
  if (jit::getAllOperatorsFor(Symbol::fromQualString("fwd::lrn")).empty()) {
    LOG(WARNING) << "fwd::lrn has not been registered in torch.";
  }

  const std::string src_pattern = R"IR(
    graph(%input, %sz, %alpha, %beta, %k, %zero, %one, %two, %three,
            %minus_one, %bool_t, %bool_f, %none):
        %4 : Tensor = aten::mul(%input, %input)
        %div.1 : Tensor = aten::unsqueeze(%4, %one)
        %9 : int = aten::size(%input, %zero)
        %10 : Tensor = prim::NumToTensor(%9)
        %12 : int = aten::Int(%10)
        %14 : int = aten::Int(%10)
        %16 : int = aten::size(%input, %one)
        %17 : Tensor = prim::NumToTensor(%16)
        %19 : int = aten::Int(%17)
        %21 : int = aten::Int(%17)
        %24 : int = aten::size(%input, %two)
        %25 : Tensor = prim::NumToTensor(%24)
        %27 : int = aten::Int(%25)
        %29 : int = aten::Int(%25)
        %32 : int = aten::size(%input, %three)
        %33 : Tensor = prim::NumToTensor(%32)
        %35 : int = aten::Int(%33)
        %41 : int[] = prim::ListConstruct(%14, %one, %21, %29, %minus_one)
        %input0.1 : Tensor = aten::view(%div.1, %41)
        %44 : int[] = prim::ListConstruct(%zero, %zero,
                          %zero, %zero, %two, %two)
        %div0.1 : Tensor = aten::constant_pad_nd(%input0.1, %44, %zero)
        %48 : int[] = prim::ListConstruct(%sz, %one, %one)
        %49 : int[] = prim::ListConstruct(%one, %one, %one)
        %zeros : int[] = prim::ListConstruct(%zero, %zero, %zero)
        %avg_pool_res : Tensor = aten::avg_pool3d(%div0.1, %48, %49,
                          %zeros, %bool_f, %bool_t, %none)
        %div1.1 : Tensor = aten::squeeze(%avg_pool_res, %one)
        %62 : int[] = prim::ListConstruct(%12, %19, %27, %35)
        %div2.1 : Tensor = aten::view(%div1.1, %62)
        %66 : Tensor = aten::mul(%div2.1, %alpha)
        %68 : Tensor = aten::add(%66, %k, %one)
        %div3.1 : Tensor = aten::pow(%68, %beta)
        %res : Tensor = aten::div(%input, %div3.1)
        return (%res))IR";

  const std::string dest_pattern = R"IR(
    graph(%input, %sz, %alpha, %beta, %k, %zero, %one, %two, %three,
            %minus_one, %bool_t, %bool_f, %none):
        %res = fwd::lrn(%input, %sz, %alpha, %beta, %k,
            %zero, %one, %two, %three, %minus_one)
        return (%res))IR";

  jit::SubgraphRewriter lrn_fuser;
  lrn_fuser.RegisterRewritePattern(src_pattern, dest_pattern);
  lrn_fuser.runOnGraph(graph);
}
