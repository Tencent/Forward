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

TEST(TestOnnxModels, DynamicMin) {
  const std::string model_path = std::string(models_dir) + "onnx_models/";
  const std::string torch_file = model_path + "resnet50_dynamic.pth";
  const std::string onnx_file = model_path + "resnet50_dynamic.onnx";

  c10::IValue input = torch::randn({1, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float32");
  // TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float16", 5e-2);
}

TEST(TestOnnxModels, DynamicOpt) {
  const std::string model_path = std::string(models_dir) + "onnx_models/";
  const std::string torch_file = model_path + "resnet50_dynamic.pth";
  const std::string onnx_file = model_path + "resnet50_dynamic.onnx";

  c10::IValue input = torch::randn({16, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float32");
  // TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float16", 5e-2);
}

TEST(TestOnnxModels, DynamicMax) {
  const std::string model_path = std::string(models_dir) + "onnx_models/";
  const std::string torch_file = model_path + "resnet50_dynamic.pth";
  const std::string onnx_file = model_path + "resnet50_dynamic.onnx";

  c10::IValue input = torch::randn({32, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float32");
  // TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float16", 5e-2);
}

TEST(TestOnnxModels, DynamicRandnA) {
  const std::string model_path = std::string(models_dir) + "onnx_models/";
  const std::string torch_file = model_path + "resnet50_dynamic.pth";
  const std::string onnx_file = model_path + "resnet50_dynamic.onnx";

  c10::IValue input = torch::randn({10, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float32");
  // TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float16", 5e-2);
}

TEST(TestOnnxModels, DynamicRandnB) {
  const std::string model_path = std::string(models_dir) + "onnx_models/";
  const std::string torch_file = model_path + "resnet50_dynamic.pth";
  const std::string onnx_file = model_path + "resnet50_dynamic.onnx";

  c10::IValue input = torch::randn({20, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float32");
  // TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float16", 5e-2);
}

TEST(TestOnnxModels, DynamicRandnC) {
  const std::string model_path = std::string(models_dir) + "onnx_models/";
  const std::string torch_file = model_path + "resnet50_dynamic.pth";
  const std::string onnx_file = model_path + "resnet50_dynamic.onnx";

  c10::IValue input = torch::randn({30, 3, 224, 224}, device);
  std::vector<c10::IValue> inputs{input};

  TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float32");
  // TestOnnxInferenceDynamic(torch_file, onnx_file, inputs, "float16", 5e-2);
}
