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

TEST(TestTorchTupleInputs, Conv2DNetwork) {
  const std::string& model_path = std::string(torch_root_dir) + "model_tuple_inputs.pth";
  const std::vector<c10::IValue> input1{::torch::randn({1, 3, 2, 2}, device),
                                        ::torch::randn({1, 2, 2, 2}, device),
                                        ::torch::randn({1, 3, 3, 3}, device)};
  const c10::IValue argument_1 = c10::ivalue::Tuple::create(input1);
  const std::vector<c10::IValue> input2{
      ::torch::randn({1, 2, 3, 3}, device),
      ::torch::randn({1, 4, 4, 4}, device),
  };
  const c10::IValue argument_2 = c10::ivalue::Tuple::create(input2);
  const at::Tensor x = ::torch::randn({1, 8, 1, 1}, device);

  const std::vector<c10::IValue> inputs = {argument_1, argument_2, x};

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = argument_1;
  input_map["argument_2"] = argument_2;
  input_map["x"] = x;

  TestTorchInference(model_path, input_map, "float32");
  // TestTorchInference(model_path, {inputs}, "float32");
}
