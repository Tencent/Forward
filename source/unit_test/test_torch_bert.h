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

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <vector>

#include "unit_test/test_batch_stream.h"
#include "unit_test/unit_test.h"

class TestTorchBert : public ::testing::Test {
 public:
  TestTorchBert() {
    model_path = std::string(torch_root_dir) + "../../models/bert_test.pt";

    batch_size = 16;
    seq_len = 50;
    max_input_id = 30522;
    const auto input_ids =
        torch::randint(max_input_id, {batch_size, seq_len}, device).toType(c10::kLong);
    const auto input_mask = torch::ones({batch_size, seq_len}, device).toType(c10::kLong);
    const auto segment_ids = torch::randint(2, {batch_size, seq_len}, device).toType(c10::kLong);
    for (int b = 0; b < batch_size; ++b) {
      for (int i = seq_len / 2; i < seq_len; i++) input_mask[b][i] = 0;
    }

    const int param = 150;
    const auto adj = torch::zeros({batch_size, param, param}).toType(c10::kFloat);
    const auto nmp = torch::zeros({batch_size, seq_len, param}).toType(c10::kFloat);

    input_map["input_ids"] = input_ids;
    input_map["attention_mask"] = input_mask;
    input_map["input"] = segment_ids;
    // input_map["adj"] = adj;
    // input_map["nmp"] = nmp;
    inputs.emplace_back(input_ids);
    inputs.emplace_back(input_mask);
    inputs.emplace_back(segment_ids);
    // inputs.emplace_back(adj);
    // inputs.emplace_back(nmp);
  }

  ~TestTorchBert() override = default;

 protected:
  void SetUp() override {}
  void TearDown() override {}

  std::string model_path;
  int seq_len = 50;
  int batch_size = 16;
  int max_input_id = 30522;
  std::vector<c10::IValue> inputs;
  std::unordered_map<std::string, c10::IValue> input_map;

 private:
  void TestBody() override {}
};

TEST_F(TestTorchBert, Float) { TestTorchInference(model_path, input_map, "float32", 5e-2); }

TEST_F(TestTorchBert, Half) { TestTorchInference(model_path, input_map, "float16", 1e-1); }

TEST_F(TestTorchBert, Int8) {
  TestTorchInference(model_path, input_map, "int8_calib", 1e-1,
                     std::make_shared<TestBertStream>(1, seq_len, max_input_id));
  TestTorchInference(model_path, input_map, "int8", 1e-1,
                     std::make_shared<TestBertStream>(1, seq_len, max_input_id));
}