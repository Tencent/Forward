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

#include "unit_test/unit_test_tf_helper.h"

class TestTfVisions : public ::testing::Test {
 protected:
  void SetUp() override {
    filename = std::string(tf_root_dir) + "../../models/tf_vision_models/";
    // configuration
    infer_mode = "float32";
    threshold = 1e-2;
  };
  void TearDown() override{};
  float threshold{1e-3};
  std::string filename;
  std::string infer_mode;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, TF_Tensor*> input_map;
};

TEST_F(TestTfVisions, DenseNet201) {
  filename = filename + "densenet201.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_4"] = input.get();
  output_names = {"relu/Relu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, InceptionV3) {
  filename = filename + "inception_v3.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_1"] = input.get();
  output_names = {"mixed10/concat"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, InceptionResNetV2) {
  filename = filename + "inception_resnet_v2.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_7"] = input.get();
  output_names = {"conv_7b_ac/Relu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, MobileNetV2) {
  filename = filename + "mobilenet_v2.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_5"] = input.get();
  output_names = {"out_relu/Relu6"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, NasNetLarge) {
  filename = filename + "nasnet_large.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 331, 331, 3});

  input_map["input_5"] = input.get();
  output_names = {"activation_556/Relu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, ResNet152V2) {
  filename = filename + "resnet152_v2.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_3"] = input.get();
  output_names = {"post_relu/Relu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, Vgg19) {
  filename = filename + "vgg19.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_2"] = input.get();
  output_names = {"block5_pool/MaxPool"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfVisions, Xception) {
  filename = filename + "xception.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_1"] = input.get();
  output_names = {"block14_sepconv2_act/Relu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}
