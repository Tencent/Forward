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

#include "torch_engine.h"

/// Test Forward-Torch.
/// This usage depends on headers below:
///   1. torch_engine.h : Interface for TfEngine and TfBuilder
///   2. common/common_macros.h : common macros for Forward

/// User should update :
///   1. model_path
///   2. Dummy Input: including data_type, dimensions and input name
int main() {
  fwd::TorchBuilder torch_builder;

  ////////////  Set Model Path  ////////////
  // Update Step 1: Update the path to pb model
  std::string model_path = "../data/softmax.pth";
  const std::string infer_mode = "float32";  // float32 / float16 / int8_calib / int8
  torch_builder.SetInferMode(infer_mode);

  ////////////  Prepare Inputs  ////////////
  // Update Step2: Update input infomations
  const c10::DeviceType device = c10::kCPU;  // c10::kCPU / c10::kCUDA
  const std::vector<int64_t> input_shape{1, 3, 7, 7};
  // DataTypes and Dimensions of dummy_input should be the same as those of real inputs
  std::vector<c10::IValue> dummy_inputs{torch::randn(input_shape, device)};
  // // build with input names
  //
  // std::unordered_map<std::string, c10::IValue> dummy_inputs;
  // // input names can be viewed by model viewers like Netron
  // dummy_inputs["input"] =  torch::randn({1, 3, 224, 224}, device);
  //

  ////////////  Build Engine  ////////////
  std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);

  std::vector<torch::jit::IValue> inputs = dummy_inputs;
  bool need_save = true;
  std::vector<torch::Tensor> outputs;
  if (!need_save) {
    ////////////  Forward  ////////////
    outputs = torch_engine->Forward(inputs);
    // outputs = torch_engine->ForwardWithName(dummy_inputs);
  } else {
    ////////////  Save, Load and Forward  ////////////
    std::string engine_file = model_path + ".engine";
    torch_engine->Save(engine_file);

    fwd::TorchEngine torch_engine;
    torch_engine.Load(engine_file);
    outputs = torch_engine.Forward(inputs);
    // outputs = torch_engine.ForwardWithName(dummy_inputs);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i] = outputs[i].cpu().contiguous().to(c10::kFloat);

    const float* res_ptr = static_cast<float*>(outputs[i].data_ptr());

    for (size_t j = 0; j < outputs[i].numel(); ++j) {
      std::cout << res_ptr[j] << ", ";
    }
  }

  std::cout << std::endl << "Test Forward-Torch finished." << std::endl;
  return 0;
}
