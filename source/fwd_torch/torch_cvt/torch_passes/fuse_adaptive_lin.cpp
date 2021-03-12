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

#include "fwd_torch/torch_cvt/torch_passes/fuse_adaptive_lin.h"

#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <string>

static auto ada_lin_registry = torch::RegisterOperators(
    // 注意：此处是直接注册到 torch 的库中，属于进程全局注册，需避免重复注册
    "fwd::adapt_lin", [](at::Tensor input, at::Tensor in_rho, at::Tensor ln_rho, at::Tensor eps1) {
      // TODO(Ao Li): 此处不必实现完整操作，仅为方便 LRN 层识别，
      // 但需保证输出 Tensor shape正确，因为 evalAll 时需要该 Tensor shape
      return input;
    });

void torch::pass::FuseAdaLin(std::shared_ptr<torch::jit::Graph>& graph) {
  if (jit::getAllOperatorsFor(Symbol::fromQualString("fwd::adapt_lin")).empty()) {
    LOG(WARNING) << "fwd::adapt_lin has not been registered in Torch.";
  }

  std::string src_pattern = R"IR(
    graph(%input, %ln_rho, %in_rho, %eps, %c_one, %c_two, %c_three,
            %c_true, %c_none):
        %9 : int[] = prim::ListConstruct(%c_two, %c_three)
        %in_mean.1 : Tensor = aten::mean(%input, %9, %c_true, %c_none)
        %14 : int[] = prim::ListConstruct(%c_two, %c_three)
        %in_var.1 : Tensor = aten::var(%input, %14, %c_true, %c_true)
        %21 : Tensor = aten::add(%in_var.1, %eps, %c_one)
        %24 : Tensor = aten::sub(%input, %in_mean.1, %c_one)
        %26 : Tensor = aten::sqrt(%21)
        %32 : Tensor = aten::div(%24, %26)
        %35 : int[] = prim::ListConstruct(%c_one, %c_two, %c_three)
        %ln_mean.1 : Tensor = aten::mean(%input, %35, %c_true, %c_none)
        %40 : int[] = prim::ListConstruct(%c_one, %c_two, %c_three)
        %ln_var.1 : Tensor = aten::var(%input, %40, %c_true, %c_true)
        %46 : Tensor = aten::add(%ln_var.1, %eps, %c_one)
        %49 : Tensor = aten::sub(%input, %ln_mean.1, %c_one)
        %51 : Tensor = aten::sqrt(%46)
        %57 : Tensor = aten::div(%49, %51)
        %mul_res : Tensor = aten::mul(%in_rho, %32)
        %mul_res_2 : Tensor = aten::mul(%ln_rho, %57)
        %res : Tensor = aten::add(%mul_res, %mul_res_2, %c_one)
        return (%res))IR";

  std::string dest_pattern = R"IR(
    graph(%input, %ln_rho, %in_rho, %eps, %c_one, %c_two, %c_three,
            %c_true, %c_none):
        %res = fwd::adapt_lin(%input, %ln_rho, %in_rho, %eps)
        return (%res))IR";

  jit::SubgraphRewriter adaptive_iln;
  adaptive_iln.RegisterRewritePattern(src_pattern, dest_pattern);
  adaptive_iln.runOnGraph(graph);
}
