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

TEST(TestTfVision, DenseNet201) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/densenet201.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_4"] = input.get();
  const std::vector<std::string> output_names{"relu/Relu"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, InceptionV3) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/inception_v3.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"mixed10/concat"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, InceptionResNetV2) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/inception_resnet_v2.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_7"] = input.get();
  const std::vector<std::string> output_names{"conv_7b_ac/Relu"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, MobileNetV2) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/mobilenet_v2.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_5"] = input.get();
  const std::vector<std::string> output_names{"out_relu/Relu6"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, NasNetLarge) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/nasnet_large.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 331, 331, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_5"] = input.get();
  const std::vector<std::string> output_names{"activation_556/Relu"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, ResNet152V2) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/resnet152_v2.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_3"] = input.get();
  const std::vector<std::string> output_names{"post_relu/Relu"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, Vgg19) {
  const std::string filename = std::string(tf_root_dir) + "../../models/tf_vision_models/vgg19.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_2"] = input.get();
  const std::vector<std::string> output_names{"block5_pool/MaxPool"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}

TEST(TestTfVision, Xception) {
  const std::string filename =
      std::string(tf_root_dir) + "../../models/tf_vision_models/xception.pb";

  const int batch_size = 1;
  const auto input =
      fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"block14_sepconv2_act/Relu"};

  TestTFInference(filename, "float32", input_map, output_names, 1e-2);
}
