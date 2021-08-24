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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <vector>

#include "common/common_macros.h"
#include "fwd_onnx/onnx_engine/onnx_engine.h"
#include "fwd_torch/torch_cvt/torch_helper.h"
#include "unit_test/unit_test.h"
#include "unit_test/unit_test_torch_helper.h"

#ifdef _MSC_VER
const char* onnx_root_dir = "../../../data/onnx_unit_tests/";
#else
const char* onnx_root_dir = "../../data/onnx_unit_tests/";
#endif

inline void TestOnnxInference(const std::string& torch_file, const std::string& onnx_file,
                              std::vector<c10::IValue>& inputs, const std::string& mode,
                              float threshold = 1e-3,
                              std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  // Calculate ground truth
  std::vector<at::Tensor> ground_truth;
  {
    // Use torch infer as onnx ground truth
    fwd::TorchInfer torch_infer;
    ASSERT_TRUE(torch_infer.LoadModel(torch_file));
    ground_truth = torch_infer.Forward(inputs, false);
  }

  const std::string engine_path = onnx_file + ".engine";

  {
    fwd::OnnxBuilder onnx_builder;
    ASSERT_TRUE(onnx_builder.SetInferMode(mode));
    if (mode.find("int8") != std::string::npos) {
      if (batch_stream == nullptr) {
        batch_stream = std::make_shared<TestBatchStream>(InputVolume(inputs));
      }
      std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
          batch_stream, getFilename(onnx_file) + ".calib", "entropy");
      onnx_builder.SetCalibrator(calib);
    }
    const auto onnx_engine = onnx_builder.Build(onnx_file);
    ASSERT_NE(onnx_engine, nullptr);
    ASSERT_TRUE(onnx_engine->Save(engine_path));
  }

  // Calculate TensorRT infer results
  std::vector<at::Tensor> results;
  {
    fwd::OnnxEngine onnx_engine;
    ASSERT_TRUE(onnx_engine.Load(engine_path));

    std::vector<fwd::Tensor> real_inputs;
    std::vector<fwd::Tensor> real_outputs;

    for (auto& input : inputs) {
      fwd::Tensor input_tensor;
      if (mode != "float32" && mode != "float") {
        if (input.toTensor().scalar_type() == c10::kFloat) {
          input = input.toTensor().to(c10::kHalf);
          input_tensor.data_type = fwd::DataType::HALF;
        }
      }
      // Take care of models like Bert
      if (input.toTensor().scalar_type() == c10::kLong ||
          input.toTensor().scalar_type() == c10::kInt) {
        input = input.toTensor().to(c10::kInt);
        input_tensor.data_type = fwd::DataType::INT32;
      }
      input_tensor.data = input.toTensor().data_ptr();
      input_tensor.dims = fwd::TrtUtils::ToVector(fwd::torch_::DimsOf(input.toTensor()));
      input_tensor.device_type = fwd::DeviceType::CPU;
      real_inputs.push_back(input_tensor);
    }

    ASSERT_TRUE(onnx_engine.Forward(real_inputs, real_outputs));
    ASSERT_TRUE(!real_outputs.empty());

    // Copy outputs back to torch tensor to ease type conversions
    auto options =
        c10::TensorOptions().layout(::torch::kStrided).requires_grad(false).device(c10::kCPU);

    for (size_t i = 0; i < real_outputs.size(); ++i) {
      const auto dtype = fwd::torch_::ToScalarType(real_outputs[i].data_type);
      options = options.dtype(dtype);

      const std::vector<int>& dims = real_outputs[i].dims;
      std::vector<int64_t> shape(dims.begin(), dims.end());
      at::Tensor output_tensor = ::torch::empty(shape, options);
      CUDA_CHECK(cudaMemcpy(output_tensor.data_ptr(), real_outputs[i].data, output_tensor.nbytes(),
                            cudaMemcpyDeviceToHost));
      results.push_back(output_tensor);
    }
  }

  ASSERT_EQ(results.size(), ground_truth.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].numel(), ground_truth[i].numel());

    // Convert to float for comparison
    results[i] = results[i].cpu().contiguous().to(c10::kFloat);
    ground_truth[i] = ground_truth[i].cpu().contiguous().to(c10::kFloat);

    const float* res_ptr = static_cast<float*>(results[i].data_ptr());
    const float* gt_ptr = static_cast<float*>(ground_truth[i].data_ptr());

    for (size_t j = 0; j < ground_truth[i].numel(); ++j) {
      ASSERT_LE(GetAbsError(res_ptr[j], gt_ptr[j]), threshold);
    }
  }
}

inline void TestOnnxInferenceDynamic(const std::string& torch_file, const std::string& onnx_file,
                                     std::vector<c10::IValue>& inputs, const std::string& mode,
                                     float threshold = 1e-3,
                                     std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  // Calculate ground truth
  std::vector<at::Tensor> ground_truth;
  {
    // Use torch infer as onnx ground truth
    fwd::TorchInfer torch_infer;
    ASSERT_TRUE(torch_infer.LoadModel(torch_file));
    ground_truth = torch_infer.Forward(inputs, false);
  }

  // For unit tests, we use ResNet50 to test dynamic inputs and manually set the Min/Opt/Max
  // sizes as 1/16/32 before generating the engine
  const std::string engine_path = (mode != "float32" && mode != "float")
                                      ? onnx_file + ".fp16" + ".engine"
                                      : onnx_file + ".engine";

  // Calculate TensorRT infer results
  std::vector<at::Tensor> results;
  {
    fwd::OnnxEngine onnx_engine;
    ASSERT_TRUE(onnx_engine.Load(engine_path));

    std::vector<fwd::Tensor> real_inputs;
    std::vector<fwd::Tensor> real_outputs;

    for (auto& input : inputs) {
      fwd::Tensor input_tensor;
      if (mode != "float32" && mode != "float") {
        if (input.toTensor().scalar_type() == c10::kFloat) {
          input = input.toTensor().to(c10::kHalf);
          input_tensor.data_type = fwd::DataType::HALF;
        }
      }
      // Take care of models like Bert
      if (input.toTensor().scalar_type() == c10::kLong ||
          input.toTensor().scalar_type() == c10::kInt) {
        input = input.toTensor().to(c10::kInt);
        input_tensor.data_type = fwd::DataType::INT32;
      }
      input_tensor.data = input.toTensor().data_ptr();
      input_tensor.dims = fwd::TrtUtils::ToVector(fwd::torch_::DimsOf(input.toTensor()));
      input_tensor.device_type = fwd::DeviceType::CPU;
      real_inputs.push_back(input_tensor);
    }

    ASSERT_TRUE(onnx_engine.Forward(real_inputs, real_outputs));
    ASSERT_TRUE(!real_outputs.empty());

    // Copy outputs back to torch tensor to ease type conversions
    auto options =
        c10::TensorOptions().layout(::torch::kStrided).requires_grad(false).device(c10::kCPU);

    for (size_t i = 0; i < real_outputs.size(); ++i) {
      const auto dtype = fwd::torch_::ToScalarType(real_outputs[i].data_type);
      options = options.dtype(dtype);

      const std::vector<int>& dims = real_outputs[i].dims;
      std::vector<int64_t> shape(dims.begin(), dims.end());
      at::Tensor output_tensor = ::torch::empty(shape, options);
      CUDA_CHECK(cudaMemcpy(output_tensor.data_ptr(), real_outputs[i].data, output_tensor.nbytes(),
                            cudaMemcpyDeviceToHost));
      results.push_back(output_tensor);
    }
  }

  ASSERT_EQ(results.size(), ground_truth.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].numel(), ground_truth[i].numel());

    // Convert to float for comparison
    results[i] = results[i].cpu().contiguous().to(c10::kFloat);
    ground_truth[i] = ground_truth[i].cpu().contiguous().to(c10::kFloat);

    const float* res_ptr = static_cast<float*>(results[i].data_ptr());
    const float* gt_ptr = static_cast<float*>(ground_truth[i].data_ptr());

    for (size_t j = 0; j < ground_truth[i].numel(); ++j) {
      ASSERT_LE(GetAbsError(res_ptr[j], gt_ptr[j]), threshold);
    }
  }
}
