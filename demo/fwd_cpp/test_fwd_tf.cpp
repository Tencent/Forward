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

#include "tf_engine.h"
#include "tf_helper.h"

/// Test Forward-Tf.
/// This usage depends on headers below:
///   1. tf_helper.h : Helper functions for TF_Tensor
///   2. tf_engine.h : Interface for TfEngine and TfBuilder
///   3. common/common_macros.h : common macros for Forward
///   4. common/fwd_utils.h : Utils for Half-Float convertion

/// User should update :
///   1. model_path
///   2. Dummy Input: including data_type, dimensions and input name
int main() {
  fwd::TfBuilder tf_builder;

  ////////////  Set Model Path  ////////////
  // Update Step 1: Update the path to pb model
  std::string model_path = "../data/softmax.pb";
  const std::string infer_mode = "float32";  // float32 / float16 / int8_calib / int8
  tf_builder.SetInferMode(infer_mode);

  ////////////  Prepare Inputs  ////////////
  // Update Step2: Update input infomations
  std::unordered_map<std::string, TF_Tensor*> input_map;
  //   Hints: DataTypes and Dimensions of dummy_input should be the same as those of real inputs
  const std::vector<int64_t> input_shape{16, 12, 24, 3};
  const auto input = CreateRandomTensor<float>(TF_FLOAT, input_shape);
  input_map["input_11"] = input.get();

  ////////////  Build Engine  ////////////
  // // build with input names
  std::shared_ptr<fwd::TfEngine> tf_engine = tf_builder.Build(model_path, input_map);

  bool need_save = true;
  std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> outputs;
  if (!need_save) {
    ////////////  Forward  ////////////
    outputs = tf_engine->ForwardWithName(input_map);
  } else {
    ////////////  Save, Load and Forward  ////////////
    std::string engine_file = model_path + ".engine";
    tf_engine->Save(engine_file);

    fwd::TfEngine tf_engine;
    tf_engine.Load(engine_file);
    outputs = tf_engine.ForwardWithName(input_map);
  }

  for (auto& output : outputs) {
    std::cout << "output name = " << output.first << ", data = " << std::endl;
    auto f_output = CastTensor<float>(output.second.get(), TF_FLOAT);
    auto data = GetTensorData<float>(f_output.get());
    for (auto ele : data) {
      std::cout << ele << ", ";
    }
  }

  std::cout << std::endl << "Test Forward-Tf finished." << std::endl;
  return 0;
}
