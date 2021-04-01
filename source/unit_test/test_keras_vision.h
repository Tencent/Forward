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

#include "unit_test/unit_test.h"

TEST(TestKerasVision, DenseNet201) {
  const std::string pb_path =
      std::string(tf_root_dir) + "../../models/tf_vision_models/densenet201.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/densenet201.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_4"}, {"relu/Relu"}, batch_size,
                     1e-4);
}

TEST(TestKerasVision, InceptionV3) {
  const std::string pb_path =
      std::string(tf_root_dir) + "../../models/tf_vision_models/inception_v3.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/inception_v3.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"}, {"mixed10/concat"},
                     batch_size, 1e-4);
}

// TEST(TestKerasVision, InceptionResNetV2) {
//   const std::string pb_path =
//       std::string(tf_root_dir) + "../../models/tf_vision_models/inception_resnet_v2.pb";
//   const std::string keras_h5_path =
//       keras_root_dir +
//       "../../models/keras_vision_models/inception_resnet_v2.h5";

//   const int batch_size = 1;
//   const auto input =
//       fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224,
//       224, 3});

//   TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_7"},
//                      {"conv_7b_ac/Relu"}, batch_size, 1e-4);
// }

TEST(TestKerasVision, MobileNetV2) {
  const std::string pb_path =
      std::string(tf_root_dir) + "../../models/tf_vision_models/mobilenet_v2.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/mobilenet_v2.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_5"}, {"out_relu/Relu6"},
                     batch_size, 2e-4);
}

TEST(TestKerasVision, NasNetLarge) {
  const std::string pb_path =
      std::string(tf_root_dir) + "../../models/tf_vision_models/nasnet_large.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/nasnet_large.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 331, 331, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_8"}, {"activation_556/Relu"},
                     batch_size, 1e-3);
}

TEST(TestKerasVision, ResNet152V2) {
  const std::string pb_path =
      std::string(tf_root_dir) + "../../models/tf_vision_models/resnet152_v2.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/resnet152_v2.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_3"}, {"post_relu/Relu"},
                     batch_size, 1e-4);
}

TEST(TestKerasVision, Vgg19) {
  const std::string pb_path = std::string(tf_root_dir) + "../../models/tf_vision_models/vgg19.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/vgg19.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_2"}, {"block5_pool/MaxPool"},
                     batch_size, 1e-4);
}

TEST(TestKerasVision, Xception) {
  const std::string pb_path =
      std::string(tf_root_dir) + "../../models/tf_vision_models/xception.pb";
  const std::string keras_h5_path =
      std::string(keras_root_dir) + "../../models/keras_vision_models/xception.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 224, 224, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"block14_sepconv2_act/Relu"}, batch_size, 1e-4);
}
