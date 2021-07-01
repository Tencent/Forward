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

#include "fwd_tf/tf_cvt/tf_utils.h"
#include "fwd_tf/tf_engine/tf_engine.h"
#include "fwd_tf/tf_engine/tf_infer.h"
#include "unit_test/unit_test.h"

#ifdef _MSC_VER
const char* tf_root_dir = "../../../data/tf_unit_tests/";
#else
const char* tf_root_dir = "../../data/tf_unit_tests/";
#endif

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

/**
 * \brief 计算输入尺寸
 */
inline std::vector<int64_t> InputVolume(
    const std::unordered_map<std::string, TF_Tensor*>& input_map) {
  std::vector<int64_t> inputSize;
  for (auto& input : input_map) {
    auto dims = fwd::tf_::GetTensorShape(input.second);
    inputSize.push_back(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>()));
  }
  return inputSize;
}

inline void TestTfTime(const std::string& graph_path,
                       std::unordered_map<std::string, TF_Tensor*>& input_map,
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

  std::vector<std::shared_ptr<TF_Tensor>> input_buffers;
  if (mode != "float32" && mode != "float") {
    for (auto& entry : input_map) {
      if (TF_TensorType(entry.second) == TF_FLOAT) {
        auto new_input = fwd::tf_::CastTensor<int16_t>(entry.second, TF_HALF);
        input_buffers.push_back(new_input);
        entry.second = new_input.get();
      }
    }
  }
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
  std::cout << "Test Iteration = " << test_count << std::endl;
  std::cout << "infer time = " << duration.count() / test_count << std::endl;
}

inline void TestTFInference(const std::string& graph_path, const std::string& mode,
                            std::unordered_map<std::string, TF_Tensor*>& input_map,
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

      if (batch_stream == nullptr) {
        batch_stream = std::make_shared<TestBatchStream>(InputVolume(input_map));
      }

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

      std::vector<std::shared_ptr<TF_Tensor>> input_buffers;
      if (mode != "float32" && mode != "float") {
        for (auto& entry : input_map) {
          if (TF_TensorType(entry.second) == TF_FLOAT) {
            auto new_input = fwd::tf_::CastTensor<int16_t>(entry.second, TF_HALF);
            input_buffers.push_back(new_input);
            entry.second = new_input.get();
          }
        }
      }

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
      ASSERT_LE(GetAbsError(results[i][j], gt_res[i][j]), threshold);
    }
  }
}
