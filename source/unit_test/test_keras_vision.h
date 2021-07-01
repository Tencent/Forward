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

#include "unit_test/unit_test_keras_helper.h"

class TestKerasVisions : public ::testing::Test {
 protected:
  void SetUp() override {
    pb_path = std::string(tf_root_dir) + "../../models/tf_vision_models/";
    keras_h5_path = std::string(keras_root_dir) + "../../models/keras_vision_models/";
    // configuration
    infer_mode = "float32";
    threshold = 1e-3;
  };
  void TearDown() override{};
  float threshold{1e-3};
  std::string pb_path;
  std::string keras_h5_path;
  std::string infer_mode;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, TF_Tensor*> input_map;
};

TEST_F(TestKerasVisions, DenseNet201) {
  pb_path = pb_path + "densenet201.pb";
  keras_h5_path = keras_h5_path + "densenet201.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_4"] = input.get();
  output_names = {"relu/Relu"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasVisions, InceptionV3) {
  pb_path = pb_path + "inception_v3.pb";
  keras_h5_path = keras_h5_path + "inception_v3.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_1"] = input.get();
  output_names = {"mixed10/concat"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasVisions, MobileNetV2) {
  pb_path = pb_path + "mobilenet_v2.pb";
  keras_h5_path = keras_h5_path + "mobilenet_v2.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_5"] = input.get();
  output_names = {"out_relu/Relu6"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasVisions, NasNetLarge) {
  pb_path = pb_path + "nasnet_large.pb";
  keras_h5_path = keras_h5_path + "nasnet_large.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 331, 331, 3});

  input_map["input_8"] = input.get();
  output_names = {"activation_556/Relu"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasVisions, ResNet152V2) {
  pb_path = pb_path + "resnet152_v2.pb";
  keras_h5_path = keras_h5_path + "resnet152_v2.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_3"] = input.get();
  output_names = {"post_relu/Relu"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasVisions, Vgg19) {
  pb_path = pb_path + "vgg19.pb";
  keras_h5_path = keras_h5_path + "vgg19.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_2"] = input.get();
  output_names = {"block5_pool/MaxPool"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasVisions, Xception) {
  pb_path = pb_path + "xception.pb";
  keras_h5_path = keras_h5_path + "xception.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  input_map["input_1"] = input.get();
  output_names = {"block14_sepconv2_act/Relu"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}
