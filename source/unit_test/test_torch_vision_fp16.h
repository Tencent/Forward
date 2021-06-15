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

TEST(TestTorchVisionFp16, AlexNet) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/alexnet.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, GoogLeNet) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/googlenet.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, Inception_v3) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/inception_v3.pth";
  const auto input = torch::randn({1, 3, 299, 299}, device);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, ResNet50) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/resnet50.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, WideResNet50_2) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/wide_resnet50_2.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, DenseNet121) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/densenet121.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, MNASNet0_75) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/mnasnet0_75.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, Mobilenet_v2) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/mobilenet_v2.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, ShuffleNet_v2_x1_5) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/shufflenet_v2_x1_5.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, SqueezeNet1_1) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/squeezenet1_1.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, VGG11_bn) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/vgg11_bn.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "float16", 0.05);
}

TEST(TestTorchVisionFp16, EfficientNet) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/effnet.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  TestTorchInference(model_path, input_map, "float16", 0.05);
}
