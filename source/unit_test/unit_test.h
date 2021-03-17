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

#include <easylogging++.h>
#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "unit_test/test_batch_stream.h"

#ifdef ENABLE_TORCH

#include "fwd_torch/torch_engine/torch_engine.h"
#include "fwd_torch/torch_engine/torch_infer.h"
#include "fwd_torch/torch_version.h"

#ifdef _MSC_VER
const c10::DeviceType device = c10::kCPU;
#else
const c10::DeviceType device = c10::kCPU;
#endif  // _MSC_VER

#endif  // ENABLE_TORCH

#ifdef ENABLE_TENSORFLOW

#include "fwd_tf/tf_cvt/tf_utils.h"
#include "fwd_tf/tf_engine/tf_engine.h"

#endif  // ENABLE_TENSORFLOW

#ifdef ENABLE_KERAS

#include "common/common_macros.h"
#include "fwd_keras/keras_engine/keras_engine.h"

#endif  // ENABLE_KERAS

#if defined(ENABLE_TENSORFLOW) || defined(ENABLE_KERAS)
#include "fwd_tf/tf_engine/tf_infer.h"
#endif  // ENABLE_TENSORFLOW || ENABLE_KERAS

#ifdef _MSC_VER
const char* torch_root_dir = "../../../data/torch_unit_tests/";
const char* tf_root_dir = "../../../data/tf_unit_tests/";
const char* keras_root_dir = "../../../data/keras_unit_tests/";
const char* models_dir = "../../../models/";
#else
const char* torch_root_dir = "../../data/torch_unit_tests/";
const char* tf_root_dir = "../../data/tf_unit_tests/";
const char* keras_root_dir = "../../data/keras_unit_tests/";
const char* models_dir = "../../models/";
#endif

template <typename T>
double GetRelativeError(T res, T gt) {
  double abs_err = static_cast<double>(abs(res - gt));
  if (abs_err < 1e-5) return 1e-5;
  return abs_err / static_cast<double>(abs(gt));
}

#ifdef ENABLE_TORCH

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
inline void TestTorchInference(const std::string& model_file,
                               const std::vector<c10::IValue>& inputs, const std::string& mode,
                               float threshold = 1e-3,
                               std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  std::vector<at::Tensor> ground_truth;
  {
    fwd::TorchInfer torch_infer;
    ASSERT_TRUE(torch_infer.LoadModel(model_file));
    {
      UTILS_PROFILE(WarmUp);
      auto _ = torch_infer.Forward(inputs, device != c10::kCPU);
    }
    ground_truth = torch_infer.Forward(inputs, device != c10::kCPU);
  }

  const std::string engine_path = model_file + ".engine";

  {
    fwd::TorchBuilder torch_builder;
    ASSERT_TRUE(torch_builder.SetInferMode(mode));
    if (mode.find("int8") != std::string::npos) {
      std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
          batch_stream, getFilename(model_file) + ".calib", "entropy");
      torch_builder.SetCalibrator(calib);
    }
    const auto torch_engine = torch_builder.Build(model_file, inputs);
    ASSERT_NE(torch_engine, nullptr);
    ASSERT_TRUE(torch_engine->Save(engine_path));
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
      ASSERT_LE(GetRelativeError(res_ptr[j], gt_ptr[j]), threshold);
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
                               const std::unordered_map<std::string, c10::IValue>& input_map,
                               const std::string& mode, float threshold = 1e-3,
                               std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  std::vector<at::Tensor> ground_truth;
  {
    fwd::TorchInfer torch_infer;
    ASSERT_TRUE(torch_infer.LoadModel(model_file));
    ground_truth = torch_infer.Forward(input_map, device != c10::kCPU);
  }

  const std::string engine_path = model_file + ".engine";

  {
    fwd::TorchBuilder torch_builder;
    ASSERT_TRUE(torch_builder.SetInferMode(mode));
    if (mode.find("int8") != std::string::npos) {
      std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
          batch_stream, getFilename(model_file) + ".calib", "entropy");
      torch_builder.SetCalibrator(calib);
    }
    const auto torch_engine = torch_builder.Build(model_file, input_map);
    ASSERT_NE(torch_engine, nullptr);
    ASSERT_TRUE(torch_engine->Save(engine_path));
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
      ASSERT_LE(GetRelativeError(res_ptr[j], gt_ptr[j]), threshold);
    }
  }
}

inline void TestTorchTime(const std::string& model_path,
                          const std::unordered_map<std::string, c10::IValue>& inputs,
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
  // GPU cold start
  auto outputs = engine->ForwardWithName(inputs);

  auto start = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < test_count; ++t) {
    auto output = engine->ForwardWithName(inputs);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;
  std::cout << "average time = " << duration.count() / test_count << std::endl;
}
#endif  // ENABLE_TORCH

#ifdef ENABLE_TENSORFLOW

/**
 * \brief 计算输入尺寸
 */
inline std::vector<int64_t> InputVolume(const std::vector<std::shared_ptr<TF_Tensor>>& inputs) {
  std::vector<int64_t> inputSize;
  for (int i = 0; i < inputs.size(); ++i) {
    auto dims = fwd::tf_::GetTensorShape(inputs[i].get());
    inputSize.push_back(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>()));
  }
  return inputSize;
}

inline void TestTfTime(const std::string& graph_path,
                       const std::unordered_map<std::string, TF_Tensor*>& input_map,
                       const std::string& mode, int test_count = 100,
                       std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  const std::string engine_path = graph_path + ".engine";

  fwd::TfBuilder builder;
  ASSERT_TRUE(builder.SetInferMode(mode));

  if (mode.find("int8") != std::string::npos) {
    std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
        // batch_stream, getFilename(graph_path) + ".calib", "entropy");
        batch_stream, getFilename(graph_path) + ".calib", "minmax");
    builder.SetCalibrator(calib);
  }
  auto engine = builder.Build(graph_path, input_map);
  ASSERT_NE(engine, nullptr);
  ASSERT_TRUE(engine->Save(engine_path));

  fwd::TfEngine tf_engine;
  ASSERT_TRUE(tf_engine.Load(engine_path));

  std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> outputs;

  // GPU cold start
  outputs = tf_engine.ForwardWithName(input_map);
  std::cout << "warm up once" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < test_count; ++i) {
    outputs = tf_engine.ForwardWithName(input_map);
  }

  ASSERT_FALSE(outputs.empty());

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration = end - start;
  std::cout << "infer time = " << duration.count() / test_count << std::endl;
}

inline void TestTFInference(const std::string& graph_path, const std::string& mode,
                            const std::unordered_map<std::string, TF_Tensor*>& input_map,
                            const std::vector<std::string>& output_names, float threshold = 1e-3,
                            std::shared_ptr<fwd::IBatchStream> batch_stream = nullptr) {
  // calculate ground truth
  std::vector<std::vector<float>> gt_res;
  {
    fwd::TFInfer tf_infer;
    ASSERT_TRUE(tf_infer.LoadGraph(graph_path));
    std::vector<std::shared_ptr<TF_Tensor>> outputs;
    ASSERT_TRUE(tf_infer.Forward(input_map, output_names, outputs));
    gt_res = fwd::tf_::GetTensorsData<float>(outputs);
  }

  const std::string engine_path = graph_path + ".engine";
  // calculate TensorRT infer results
  std::vector<std::vector<float>> results;
  {
    {
      fwd::TfBuilder tf_builder;
      ASSERT_TRUE(tf_builder.SetInferMode(mode));

      if (mode.find("int8") != std::string::npos) {
        std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
            // batch_stream, getFilename(graph_path) + ".calib", "entropy");
            batch_stream, getFilename(graph_path) + ".calib", "minmax");
        tf_builder.SetCalibrator(calib);
      }
      auto tf_engine = tf_builder.Build(graph_path, input_map);
      ASSERT_NE(tf_engine, nullptr);
      ASSERT_TRUE(tf_engine->Save(engine_path));
    }

    std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> outputs;
    {
      fwd::TfEngine tf_engine;
      ASSERT_TRUE(tf_engine.Load(engine_path));
      outputs = tf_engine.ForwardWithName(input_map);
      for (auto& output : outputs) {
        output.second = fwd::tf_::CastTensor<float>(output.second.get(), TF_FLOAT);
      }
    }

    for (auto& output : outputs) {
      results.push_back(fwd::tf_::GetTensorData<float>(output.second.get()));
    }
  }

  ASSERT_EQ(gt_res.size(), results.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(gt_res[i].size(), results[i].size());
    for (size_t j = 0; j < results[i].size(); ++j) {
      ASSERT_LE(GetRelativeError(results[i][j], gt_res[i][j]), threshold);
    }
  }
}

#endif  // ENABLE_TENSORFLOW

#ifdef ENABLE_KERAS

inline void TestKerasInference(const std::string& pb_path, const std::string& keras_h5_path,
                               const std::vector<TF_Tensor*>& inputs,
                               const std::vector<std::string>& input_names,
                               const std::vector<std::string>& output_names, int batch_size = 1,
                               float threshold = 1e-3) {
  std::unordered_map<std::string, TF_Tensor*> input_map;
  for (int i = 0; i < input_names.size(); ++i) {
    input_map[input_names[i]] = inputs[i];
  }

  // calculate ground truth
  std::vector<std::vector<float>> ground_truth;
  {
    // TODO(Ao Li): 使用 tensorflow pb 作为 keras groundtruth
    fwd::TFInfer tf_infer;
    if (!tf_infer.LoadGraph(pb_path)) {
      ASSERT_TRUE(false);
    }
    std::vector<std::shared_ptr<TF_Tensor>> outputs;
    if (!tf_infer.Forward(input_map, output_names, outputs)) {
      ASSERT_TRUE(false);
    }
    // TODO(Ao Li): 这里假设输出一定是 float 类型
    ground_truth = fwd::tf_::GetTensorsData<float>(outputs);
  }

  // calculate TensorRT infer results
  std::vector<std::vector<float>> results;
  {
    std::shared_ptr<fwd::KerasEngine> keras_engine;
    {
      fwd::KerasBuilder keras_builder;
      // ASSERT_TRUE(keras_builder.SetInferMode(mode));

      keras_engine = keras_builder.Build(keras_h5_path, batch_size);
    }
    ASSERT_TRUE(keras_engine != nullptr);

    // keras inputs
    std::vector<void*> input_buffers;
    std::vector<std::vector<int>> input_dims;
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_buffers.push_back(TF_TensorData(inputs[i]));
      std::vector<int> input_shape(TF_NumDims(inputs[i]));
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_shape[j] = TF_Dim(inputs[i], j);
      }
      input_dims.push_back(input_shape);
    }

    // outputs
    std::vector<void*> output_buffers;
    std::vector<std::vector<int>> output_dims;
    ASSERT_TRUE(
        keras_engine->Forward(input_buffers, input_dims, output_buffers, output_dims, false));
    ASSERT_EQ(output_buffers.size(), output_dims.size());

    // copy outputs back
    for (size_t i = 0; i < output_dims.size(); ++i) {
      const auto count = fwd::TrtUtils::Volume(output_dims[i]);
      ASSERT_TRUE(output_buffers[i]);
      std::vector<float> result(count);
      CUDA_CHECK(cudaMemcpy(result.data(), output_buffers[i], count * sizeof(float),
                            cudaMemcpyDeviceToHost));
      results.push_back(result);
    }
  }

  ASSERT_EQ(ground_truth.size(), results.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(ground_truth[i].size(), results[i].size());
    for (size_t j = 0; j < results[i].size(); ++j) {
      ASSERT_LE(GetRelativeError(results[i][j], ground_truth[i][j]), threshold);
    }
  }
}

#endif  // ENABLE_KERAS
