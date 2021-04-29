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

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

#include "cuda_helper.h"
#include "trt_fwd_engine.h"

/// Test Forward-Trt.
/// This usage depends on headers below:
///   1. cuda_helper.h : Device memory manipulate handlers
///   2. trt_fwd_engine.h : Interface for TrtForwardEngine
///   3. common/i_forward_api.h : Abstract interface for TrtForwardEngine
///   4. common/common_macros.h : common macros for Forward
///   5. common/fwd_common.h : Common utils for Forward
///   6. common/trt_common.h : Common utils for TensorRT

/// User should update :
///   1. model_path
///   2. Dummy Input: including data_type, dimensions and input name
int main() {
  fwd::TrtForwardEngine fwd_engine;

  // Update Step 1: Update the path to pb model
  std::string engine_path = "../data/softmax.pb.engine";

  ////////////  Load Engine  ////////////
  if (!fwd_engine.Load(engine_path)) {
    std::cerr << "Engine Load failed on " << engine_path << std::endl;
    return -1;
  }

  ////////////  Prepare Inputs  ////////////
  fwd::IOMappingVector feed_dict;
  fwd::IOMappingVector outputs;
  // Update Step2: Update input infomations
  // create dummy input
  fwd::NamedTensor input;
  input.name = "input_11";                       // required
  auto shape = std::vector<int>{16, 12, 24, 3};  // required
  input.tensor.dims = shape;
  auto volume = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
  PinnedVector<float> data;
  data.Resize(volume);
  memset(data.Data(), 0, sizeof(float) * volume);
  input.tensor.data = data.Data();
  input.tensor.data_type = fwd::DataType::FLOAT;
  input.tensor.device_type = fwd::DeviceType::CPU;
  feed_dict.push_back(input);

  ////////////  Forwarding  ////////////
  if (!fwd_engine.ForwardWithName(feed_dict, outputs)) {
    std::cerr << "Engine forward error! " << std::endl;
    return -1;
  }

  ////////////  Get Outputs  ////////////
  PinnedVector<float> h_outputs;
  const auto& out_shape = outputs[0].tensor.dims;
  auto out_volume =
      std::accumulate(out_shape.cbegin(), out_shape.cend(), 1, std::multiplies<int>());
  h_outputs.Resize(out_volume);
  MemcpyDeviceToHost(h_outputs.Data(), reinterpret_cast<float*>(outputs[0].tensor.data),
                     out_volume);

  ////////////  Print Outputs  ////////////
  auto* data_ptr = h_outputs.Data();
  std::cout << "Print Head 10 elements" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << *(data_ptr + i) << ", ";
  }

  std::cout << std::endl << "Test Engine finished." << std::endl;
  return 0;
}
