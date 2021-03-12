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

#include <string>
#include <vector>

#include "unit_test/unit_test.h"

TEST(TestTorchDlrm, DLRM) {
  const std::string& model_path = std::string(torch_root_dir) + "dlrm.pth";

  std::vector<size_t> offset(6, 0);
  offset[1] = 1;
  offset[3] = 1;
  offset[5] = 1;

  std::vector<int> input0{3, 0, 2};
  std::vector<int> input1{1, 1, 2};
  std::vector<int> input2{1, 1};

  const torch::Tensor sparse_offset =
      ::torch::from_blob(offset.data(), {3, 2}, ::torch::requires_grad(false).dtype(c10::kLong))
          .clone();
  const std::vector<at::Tensor> sparse_input{
      ::torch::tensor({3, 0, 2}, ::torch::requires_grad(false).dtype(c10::kLong)),
      ::torch::tensor({1, 1, 2}, ::torch::requires_grad(false).dtype(c10::kLong)),
      ::torch::tensor({1, 1}, ::torch::requires_grad(false).dtype(c10::kLong)),
  };

  const auto dense_input = ::torch::randn({2, 4}, device);
  TestTorchInference(model_path, {dense_input, sparse_offset, sparse_input}, "float32", 1e-04);
}
