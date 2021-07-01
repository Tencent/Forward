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

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <vector>

#include "fwd_torch/torch_engine/torch_engine.h"
#include "fwd_torch/torch_engine/torch_infer.h"
#include "unit_test/unit_test.h"

#ifdef _MSC_VER
const char* torch_root_dir = "../../../data/torch_unit_tests/";
const c10::DeviceType device = c10::kCPU;
#else
const c10::DeviceType device = c10::kCUDA;
const char* torch_root_dir = "../../data/torch_unit_tests/";
#endif

/**
 * \brief 计算输入尺寸
 */
inline std::vector<int64_t> InputVolume(const std::vector<c10::IValue>& inputs) {
  std::vector<int64_t> inputSize;
  for (int i = 0; i < inputs.size(); ++i) {
    auto dims = fwd::TrtUtils::ToVector(fwd::torch_::DimsOf(inputs[i].toTensor()));
    inputSize.push_back(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>()));
  }
  return inputSize;
}

/**
 * \brief 计算输入尺寸
 */
inline std::vector<int64_t> InputVolume(
    const std::unordered_map<std::string, c10::IValue>& input_map) {
  std::vector<int64_t> inputSize;
  for (auto& entry : input_map) {
    auto dims = fwd::TrtUtils::ToVector(fwd::torch_::DimsOf(entry.second.toTensor()));
    inputSize.push_back(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>()));
  }
  return inputSize;
}

/**
 * \brief 测试单次 torch jit 和 TensorRT 推理
 * \param model_file 模型文件路径
 * \param inputs 实际输入，位于 CPU / CUDA
 * \param mode 推理模式，可以是 "float32", "float16", "int8"
 * \param threshold 浮点数误差允许的阈值
 */
inline void TestTorchInference(const std::string& model_file, std::vector<c10::IValue>& inputs,
                               const std::string& mode, float threshold = 1e-3,
                               std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  std::vector<at::Tensor> ground_truth;
  {
    fwd::TorchInfer torch_infer;
    ASSERT_TRUE(torch_infer.LoadModel(model_file));
    ground_truth = torch_infer.Forward(inputs, false);
  }

  const std::string engine_path = model_file + ".engine";

  {
    fwd::TorchBuilder torch_builder;
    ASSERT_TRUE(torch_builder.SetInferMode(mode));
    if (mode.find("int8") != std::string::npos) {
      if (batch_stream == nullptr) {
        batch_stream = std::make_shared<TestBatchStream>(InputVolume(inputs));
      }
      std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
          batch_stream, getFilename(model_file) + ".calib", "entropy");
      torch_builder.SetCalibrator(calib);
    }
    const auto torch_engine = torch_builder.Build(model_file, inputs);
    ASSERT_NE(torch_engine, nullptr);
    ASSERT_TRUE(torch_engine->Save(engine_path));
  }

  if (mode != "float32" && mode != "float") {
    for (auto& input : inputs) {
      if (input.toTensor().scalar_type() == c10::kFloat) {
        input = input.toTensor().to(c10::kHalf);
      }
    }
  }

  std::vector<at::Tensor> results;
  {
    fwd::TorchEngine torch_engine;
    ASSERT_TRUE(torch_engine.Load(engine_path));
    results = torch_engine.Forward(inputs);
  }

  ASSERT_EQ(results.size(), ground_truth.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].numel(), ground_truth[i].numel());

    // 转成 float 进行比较
    results[i] = results[i].cpu().contiguous().to(c10::kFloat);
    ground_truth[i] = ground_truth[i].cpu().contiguous().to(c10::kFloat);

    const float* res_ptr = static_cast<float*>(results[i].data_ptr());
    const float* gt_ptr = static_cast<float*>(ground_truth[i].data_ptr());

    for (size_t j = 0; j < ground_truth[i].numel(); ++j) {
      ASSERT_LE(GetAbsError(res_ptr[j], gt_ptr[j]), threshold);
    }
  }
}

/**
 * \brief 测试单次 torch jit 和 TensorRT 推理
 * \param model_file 模型文件路径
 * \param inputs 实际输入，位于 CPU / CUDA
 * \param mode 推理模式，可以是 "float32", "float16", "int8"
 * \param threshold 浮点数误差允许的阈值
 */
inline void TestTorchInference(const std::string& model_file,
                               std::unordered_map<std::string, c10::IValue>& input_map,
                               const std::string& mode, float threshold = 1e-3,
                               std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  std::vector<at::Tensor> ground_truth;
  {
    fwd::TorchInfer torch_infer;
    ASSERT_TRUE(torch_infer.LoadModel(model_file));
    ground_truth = torch_infer.Forward(input_map, false);
  }

  const std::string engine_path = model_file + ".engine";

  {
    fwd::TorchBuilder torch_builder;
    ASSERT_TRUE(torch_builder.SetInferMode(mode));
    if (mode.find("int8") != std::string::npos) {
      if (batch_stream == nullptr) {
        batch_stream = std::make_shared<TestBatchStream>(InputVolume(input_map));
      }
      std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
          batch_stream, getFilename(model_file) + ".calib", "entropy");
      torch_builder.SetCalibrator(calib);
    }
    const auto torch_engine = torch_builder.Build(model_file, input_map);
    ASSERT_NE(torch_engine, nullptr);
    ASSERT_TRUE(torch_engine->Save(engine_path));
  }

  if (mode != "float32" && mode != "float") {
    for (auto& entry : input_map) {
      auto& input = entry.second;
      if (input.toTensor().scalar_type() == c10::kFloat) {
        input = input.toTensor().to(c10::kHalf);
      }
    }
  }

  std::vector<at::Tensor> results;
  {
    fwd::TorchEngine torch_engine;
    ASSERT_TRUE(torch_engine.Load(engine_path));
    results = torch_engine.ForwardWithName(input_map);
  }

  ASSERT_EQ(results.size(), ground_truth.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].numel(), ground_truth[i].numel());

    // 转成 float 进行比较
    results[i] = results[i].cpu().contiguous().to(c10::kFloat);
    ground_truth[i] = ground_truth[i].cpu().contiguous().to(c10::kFloat);

    const float* res_ptr = static_cast<float*>(results[i].data_ptr());
    const float* gt_ptr = static_cast<float*>(ground_truth[i].data_ptr());

    for (size_t j = 0; j < ground_truth[i].numel(); ++j) {
      ASSERT_LE(GetAbsError(res_ptr[j], gt_ptr[j]), threshold);
    }
  }
}

inline void TestTorchTime(const std::string& model_path,
                          std::unordered_map<std::string, c10::IValue>& inputs,
                          const std::string& mode = "float32", int test_count = 100,
                          std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  auto builder = fwd::TorchBuilder();
  ASSERT_TRUE(builder.SetInferMode(mode));
  if (mode.find("int8") != std::string::npos) {
    std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
        // batch_stream, getFilename(graph_path) + ".calib", "entropy");
        batch_stream, getFilename(model_path) + ".calib", "minmax");
    builder.SetCalibrator(calib);
  }
  auto engine = builder.Build(model_path, inputs);

  if (mode != "float32" && mode != "float") {
    for (auto& entry : inputs) {
      auto& input = entry.second;
      if (input.toTensor().scalar_type() == c10::kFloat) {
        input = input.toTensor().to(c10::kHalf);
      }
    }
  }
  // GPU cold start
  auto outputs = engine->ForwardWithName(inputs);

  auto start = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < test_count; ++t) {
    auto output = engine->ForwardWithName(inputs);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;
  std::cout << "Test Iteration = " << test_count << std::endl;
  std::cout << "average time = " << duration.count() / test_count << std::endl;
}