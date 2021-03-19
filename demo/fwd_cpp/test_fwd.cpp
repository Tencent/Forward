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

int main() {
  fwd::TorchBuilder torch_builder;

  std::string model_path = "path/to/jit/module";
  const std::string infer_mode = "float32"; // float32 / float16 / int8_calib / int8
  const c10::DeviceType device = c10::kCPU; // c10::kCPU / c10::kCUDA
  // DataTypes and Dimensions of dummy_input should be the same as those of real inputs 
  std::vector<c10::IValue> dummy_inputs {torch::randn({1, 3, 224, 224}, device)};
  // // build with input names
  //
  // std::unordered_map<std::string, c10::IValue> dummy_inputs;
  // // input names can be viewed by model viewers like Netron
  // dummy_inputs["input"] =  torch::randn({1, 3, 224, 224}, device); 
  //

  torch_builder.SetInferMode(infer_mode);
  std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);

  std::vector<torch::jit::IValue> inputs = dummy_inputs;
  bool need_save = true;
  if (!need_save){
      std::vector<torch::Tensor> outputs = torch_engine->Forward(inputs);
      // std::vector<torch::Tensor> outputs = torch_engine->ForwardWithName(dummy_inputs); 
  }else{
      std::string engine_file = "path/to/out/engine";
      torch_engine->Save(engine_file);

      fwd::TorchEngine torch_engine;
      torch_engine.Load(engine_file);
      std::vector<torch::Tensor> outputs = torch_engine.Forward(inputs);
      // std::vector<torch::Tensor> outputs = torch_engine.ForwardWithName(dummy_inputs);
  }
  return 0;
}
