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

#include <string>

#include "unit_test/unit_test_onnx_helper.h"

class TestOnnxModels : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path = std::string(models_dir) + "onnx_models/";
    // configuration
    infer_mode = "float32";
    threshold = 1e-3;
  };
  void TearDown() override{};
  float threshold{1e-3};
  std::string model_path;
  std::string infer_mode;
};

TEST_F(TestOnnxModels, ResNet50) {
  const std::string torch_file = model_path + "resnet50.pth";
  const std::string onnx_file = model_path + "resnet50.onnx";

  c10::IValue input = torch::randn({1, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInference(torch_file, onnx_file, inputs, infer_mode);
  // TestOnnxInference(torch_file, onnx_file, inputs, "float16", 5e-2);
  // TestOnnxInference(torch_file, onnx_file, inputs, "int8", 1e-1);
}

TEST_F(TestOnnxModels, Tuple) {
  const std::string torch_file = model_path + "tuple.pth";
  const std::string onnx_file = model_path + "tuple.onnx";

  c10::IValue input1 = torch::randn({1, 3, 224, 224}, device);
  c10::IValue input2 = torch::randn({1, 3, 224, 224}, device);
  c10::IValue input3 = torch::randn({1, 3, 224, 224}, device);
  c10::IValue input4 = torch::randn({1, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input1, input2, input3, input4};

  TestOnnxInference(torch_file, onnx_file, inputs, infer_mode);
  // TestOnnxInference(torch_file, onnx_file, inputs, "float16", 5e-2);
  // TestOnnxInference(torch_file, onnx_file, inputs, "int8", 1e-1);
}

TEST_F(TestOnnxModels, LSTM) {
  const std::string torch_file = model_path + "lstm.pth";
  const std::string onnx_file = model_path + "lstm.onnx";

  c10::IValue input = torch::randn({1, 50, 128}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInference(torch_file, onnx_file, inputs, infer_mode);
  // TestOnnxInference(torch_file, onnx_file, inputs, "float16", 5e-2);
  // TestOnnxInference(torch_file, onnx_file, inputs, "int8", 1e-1);
}
