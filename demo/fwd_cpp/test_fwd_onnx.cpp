// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express
// or implied. See the License for the specific language governing permissions
// and limitations under
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

#include "common/fwd_common.h"
#include "common/trt_common.h"
#include "cuda_helper.h"
#include "onnx_engine.h"

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

/// Test Forward-Onnx.
/// This usage depends on headers below:
///   1. onnx_engine.h : Interface for TfEngine and TfBuilder
///   2. common/common_macros.h : common macros for Forward

/// User should update :
///   1. model_path
///   2. input: including data, dimensions, data type, and device type
int main() {
  fwd::OnnxBuilder onnx_builder;

  ////////////  Set Model Path  ////////////
  // Update Step 1: Update the path to pb model
  std::string model_path = "../../../models/onnx_models/resnet50.onnx";
  const std::string infer_mode = "float32";  // float32 / float16 / int8_calib / int8
  onnx_builder.SetInferMode(infer_mode);

  ////////////  Prepare Inputs  ////////////
  // Update Step2: Update input infomations
  const auto shape = std::vector<int>{1, 3, 224, 224};
  const auto volume = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
  std::vector<float> data;
  data.resize(volume);
  std::memset(data.data(), 0, sizeof(float) * volume);

  fwd::Tensor input;
  input.data = data.data();
  input.dims = shape;
  input.data_type = fwd::DataType::FLOAT;
  input.device_type = fwd::DeviceType::CPU;
  const std::vector<fwd::Tensor> inputs{input};

  //////////////  Build Engine  //////////////
  std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);

  //////////////  Forwarding  //////////////
  std::vector<fwd::Tensor> outputs;
  if (!onnx_engine->Forward(inputs, outputs)) {
    std::cerr << "Engine forward error! " << std::endl;
    return -1;
  }

  //////////////  Get Outputs  //////////////
  // Note: the default `outputs` is on device memory
  // Optional: copy the `outputs` to host memory
  std::vector<std::vector<float>> h_outputs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::vector<float> h_out;
    const auto &out_shape = outputs[i].dims;
    auto out_volume =
        std::accumulate(out_shape.cbegin(), out_shape.cend(), 1, std::multiplies<int>());
    h_out.resize(out_volume);
    MemcpyDeviceToHost(h_out.data(), reinterpret_cast<float *>(outputs[i].data), out_volume);
    h_outputs.push_back(std::move(h_out));
  }

  //////////////  Print Outputs  //////////////
  for (const auto &h_output : h_outputs) {
    for (const auto &out : h_output) {
      std::cout << out << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Test Forward-ONNX finished." << std::endl;
  return 0;
}
