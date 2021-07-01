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

#include "unit_test/unit_test_torch_helper.h"

class TestTorchVision : public ::testing::Test {
 protected:
  void SetUp() override {
    // configuration
    model_path = std::string(models_dir) + "torch_vision_models/";
    infer_mode = "float32";
    threshold = 1e-3;
  };
  void TearDown() override{};
  std::string model_path;
  std::string infer_mode;
  float threshold{1e-3};
  std::shared_ptr<fwd::IBatchStream> batch_stream{nullptr};
  std::unordered_map<std::string, c10::IValue> input_map;
};

TEST_F(TestTorchVision, AlexNet) {
  model_path = model_path + "alexnet.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, GoogLeNet) {
  model_path = model_path + "googlenet.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, Inception_v3) {
  model_path = model_path + "inception_v3.pth";
  input_map["input"] = torch::randn({1, 3, 299, 299}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, ResNet50) {
  model_path = model_path + "resnet50.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, WideResNet50_2) {
  model_path = model_path + "wide_resnet50_2.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, DenseNet121) {
  model_path = model_path + "densenet121.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, MNASNet0_75) {
  model_path = model_path + "mnasnet0_75.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, Mobilenet_v2) {
  model_path = model_path + "mobilenet_v2.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, ShuffleNet_v2_x1_5) {
  model_path = model_path + "shufflenet_v2_x1_5.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, SqueezeNet1_1) {
  model_path = model_path + "squeezenet1_1.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, VGG11_bn) {
  model_path = model_path + "vgg11_bn.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchVision, EfficientNet) {
  model_path = model_path + "effnet.pth";
  input_map["input"] = torch::randn({1, 3, 224, 224}, device);
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}
