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

#include <queue>
#include <set>
#include <string>
#include <vector>

#include "unit_test/unit_test.h"

#define GTEST_COUT std::cerr << "[   INFO   ] "
/**
 * \brief topk是否包含相同元素，且依次相等
 * \param res1 结果1
 * \param res2 结果2
 * \param topk 检查个数
 */
inline bool TopKCheck(const float* result1, const float* result2, int num, int topk = 5) {
  std::priority_queue<std::pair<float, size_t>> res1_queue;
  std::priority_queue<std::pair<float, size_t>> res2_queue;
  for (size_t i = 0; i < num; ++i) {
    res1_queue.push(std::pair<float, int>(result1[i], i));
    res2_queue.push(std::pair<float, int>(result2[i], i));
  }

  std::set<size_t> index1_set;
  std::set<size_t> index2_set;
  for (int i = 0; i < topk; ++i) {
    size_t index1 = res1_queue.top().second;
    size_t index2 = res2_queue.top().second;
    if (index1 != index2) {
      GTEST_COUT << "Top [" << i + 1 << "] diff: result1 got No." << index1 << " = "
                 << res1_queue.top().first << " & result2 got No." << index2 << " = "
                 << res2_queue.top().first << std::endl;
    }
    index1_set.insert(index1);
    index2_set.insert(index2);
    res1_queue.pop();
    res2_queue.pop();
  }

  int count = 0;
  for (auto it1 = index1_set.begin(), it2 = index2_set.begin(); it1 != index1_set.end();
       it1++, it2++) {
    if (*it1 != *it2) ++count;
  }
  if (count > 2) {
    GTEST_COUT << "has more than two diff classify result." << std::endl;
    return false;
  }
  return true;
}

/**
 * \brief 测试单次 torch jit 和 TensorRT **INT8** 下推理结果对比
 * \param model_file 模型文件路径
 * \param inputs 实际输入，位于 CPU / CUDA
 * \param mode 推理模式，可以是 "float32", "float16", "int8"
 * \param topk 浮点数误差允许的阈值
 */
inline void TestTorchInferenceClassify(const std::string& model_file,
                                       const std::unordered_map<std::string, c10::IValue>& inputs,
                                       const std::string& mode, float topk = 5) {
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

  {
    fwd::TorchBuilder torch_builder;
    ASSERT_TRUE(torch_builder.SetInferMode(mode));
    if (mode == "int8") {
      auto inputSize = InputVolume(inputs);
      std::shared_ptr<fwd::IBatchStream> tbs = std::make_shared<TestBatchStream>(inputSize);
      std::shared_ptr<fwd::TrtInt8Calibrator> calib = std::make_shared<fwd::TrtInt8Calibrator>(
          tbs, getFilename(model_file) + ".calib", "entropy");
      torch_builder.SetCalibrator(calib);
    }
    const auto torch_engine = torch_builder.Build(model_file, inputs);
    ASSERT_NE(torch_engine, nullptr);
    ASSERT_TRUE(torch_engine->Save("temp.engine"));
  }

  std::vector<at::Tensor> results;
  {
    fwd::TorchEngine torch_engine;
    ASSERT_TRUE(torch_engine.Load("temp.engine"));
    results = torch_engine.ForwardWithName(inputs);
  }

  ASSERT_EQ(results.size(), ground_truth.size());

  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_EQ(results[i].numel(), ground_truth[i].numel());

    results[i] = results[i].cpu().contiguous().to(c10::kFloat);
    ground_truth[i] = ground_truth[i].cpu().contiguous().to(c10::kFloat);
    const float* res_ptr = static_cast<float*>(results[i].data_ptr());
    const float* gt_ptr = static_cast<float*>(ground_truth[i].data_ptr());
    ASSERT_TRUE(TopKCheck(res_ptr, gt_ptr, results[i].numel()));
  }
}

TEST(TestTorchVisionInt8, AlexNet) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/alexnet.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, GoogLeNet) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/googlenet.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, Inception_v3) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/inception_v3.pth";
  const auto input = torch::randn({1, 3, 299, 299}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, ResNet50) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/resnet50.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, WideResNet50_2) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/wide_resnet50_2.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, DenseNet121) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/densenet121.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, MNASNet0_75) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/mnasnet0_75.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, Mobilenet_v2) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/mobilenet_v2.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, ShuffleNet_v2_x1_5) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/shufflenet_v2_x1_5.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, SqueezeNet1_1) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/squeezenet1_1.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, VGG11_bn) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/vgg11_bn.pth";
  const auto input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInferenceClassify(model_path, input_map, "int8");
}

TEST(TestTorchVisionInt8, EfficientNet) {
  const auto model_path = std::string(models_dir) + "torch_vision_models/effnet.pth";
  const c10::IValue input = torch::randn({1, 3, 224, 224}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  TestTorchInferenceClassify(model_path, input_map, "int8");
}
