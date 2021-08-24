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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <vector>

#include "unit_test/unit_test_onnx_helper.h"

class TestOnnxBert : public ::testing::Test {
 public:
  TestOnnxBert() {
    torch_file = std::string(models_dir) + "onnx_models/bert.pth";
    onnx_file = std::string(models_dir) + "onnx_models/bert.onnx";

    seq_len = 128;
    batch_size = 32;
    max_input_id = 30522;
    const auto input_ids =
        torch::randint(max_input_id, {batch_size, seq_len}, device).toType(c10::kLong);
    const auto attention_mask = torch::ones({batch_size, seq_len}, device).toType(c10::kLong);
    const auto segment_ids = torch::randint(2, {batch_size, seq_len}, device).toType(c10::kLong);
    for (int b = 0; b < batch_size; ++b) {
      for (int i = seq_len / 2; i < seq_len; i++) attention_mask[b][i] = 0;
    }

    inputs.emplace_back(input_ids);
    inputs.emplace_back(attention_mask);
    inputs.emplace_back(segment_ids);
  }

  ~TestOnnxBert() override = default;

 protected:
  void SetUp() override {}
  void TearDown() override {}

  std::string torch_file;
  std::string onnx_file;
  int seq_len = 128;
  int batch_size = 32;
  int max_input_id = 30522;
  std::vector<c10::IValue> inputs;

 private:
  void TestBody() override {}
};

TEST_F(TestOnnxBert, Float) {
  TestOnnxInference(torch_file, onnx_file, inputs, "float32", 5e-2);
}

TEST_F(TestOnnxBert, Half) {
  TestOnnxInference(torch_file, onnx_file, inputs, "float16", 1e-1);
}

TEST_F(TestOnnxBert, Int8Calib) {
  TestOnnxInference(torch_file, onnx_file, inputs, "int8_calib", 1e-1);
}

TEST_F(TestOnnxBert, Int8) {
  TestOnnxInference(torch_file, onnx_file, inputs, "int8", 1e-1);
}
