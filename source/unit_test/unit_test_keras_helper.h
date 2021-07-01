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

#include "common/common_macros.h"
#include "fwd_keras/keras_engine/keras_engine.h"
#include "unit_test/unit_test.h"
#include "unit_test/unit_test_tf_helper.h"

#ifdef _MSC_VER
const char* keras_root_dir = "../../../data/keras_unit_tests/";
#else
const char* keras_root_dir = "../../data/keras_unit_tests/";
#endif

inline void TestKerasInference(const std::string& pb_path, const std::string& keras_h5_path,
                               std::vector<std::pair<std::string, TF_Tensor*>>& input_map,
                               const std::vector<std::string>& output_names, int batch_size = 1,
                               float threshold = 1e-3) {
  // calculate ground truth
  std::vector<std::vector<float>> ground_truth;
  {
    // TODO(Ao Li): 使用 tensorflow pb 作为 keras groundtruth
    fwd::TFInfer tf_infer;
    if (!tf_infer.LoadGraph(pb_path)) {
      ASSERT_TRUE(false);
    }
    std::unordered_map<std::string, TF_Tensor*> tf_input;
    for (auto& entry : input_map) tf_input[entry.first] = entry.second;
    std::vector<std::shared_ptr<TF_Tensor>> outputs;
    if (!tf_infer.Forward(tf_input, output_names, outputs)) {
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
    std::vector<fwd::Tensor> real_inputs;
    std::vector<fwd::Tensor> real_outputs;
    for (auto& entry : input_map) {
      fwd::Tensor tensor;
      tensor.data = TF_TensorData(entry.second);
      tensor.dims = fwd::TrtUtils::ToVector(fwd::tf_::DimsOf(entry.second));
      tensor.device_type = fwd::DeviceType::CPU;
      if (TF_TensorType(entry.second) == TF_INT32) {
        tensor.data_type = fwd::DataType::INT32;
      } else if (TF_TensorType(entry.second) == TF_HALF) {
        tensor.data_type = fwd::DataType::HALF;
      }
      real_inputs.push_back(tensor);
    }

    ASSERT_TRUE(keras_engine->Forward(real_inputs, real_outputs));
    ASSERT_TRUE(!real_outputs.empty());

    // copy outputs back
    for (size_t i = 0; i < real_outputs.size(); ++i) {
      const auto count = fwd::TrtUtils::Volume(real_outputs[i].dims);
      ASSERT_TRUE(real_outputs[i].data);
      std::vector<float> result(count);
      CUDA_CHECK(cudaMemcpy(result.data(), real_outputs[i].data, count * sizeof(float),
                            cudaMemcpyDeviceToHost));
      results.push_back(result);
    }
  }

  ASSERT_EQ(ground_truth.size(), results.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(ground_truth[i].size(), results[i].size());
    for (size_t j = 0; j < results[i].size(); ++j) {
      ASSERT_LE(GetAbsError(results[i][j], ground_truth[i][j]), threshold);
    }
  }
}
