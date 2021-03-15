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

#include <string>
#include <vector>

#include "unit_test/unit_test.h"

TEST(TestTorchNodes131Int8, Arithmetic) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/arithmetic.pth";
  const auto a = ::torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);
  const auto b = ::torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["a"] = a;
  input_map["b"] = b;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Addmm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/addmm.pth";
  const auto M = ::torch::randn({3, 3}, device).to(c10::kHalf);
  const auto mat1 = ::torch::randn({3, 3}, device).to(c10::kHalf);
  const auto mat2 = ::torch::randn({3, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["M"] = M;
  input_map["mat1"] = mat1;
  input_map["mat2"] = mat2;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, FullyConnected) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/fully_connected.pth";
  const auto input = ::torch::randn({1, 10}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Activation) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/activation.pth";
  const auto input = ::torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, PRelu) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/prelu.pth";
  const auto input = torch::randn({1, 3, 11, 13}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Inplace) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/inplace.pth";
  const auto input = torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, InstanceNorm2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/instance_norm.pth";
  const auto input = torch::randn({3, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, InstanceNorm2dTrack) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/instance_norm_track.pth";
  const auto input = torch::randn({4, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, InstanceNorm2dAffine) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/instance_norm_aff.pth";
  const auto input = torch::randn({5, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, InstanceNorm2dAffineTrack) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/instance_norm_aff_track.pth";
  const auto input = torch::randn({6, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Norm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/norm.pth";
  const auto input = torch::randn({1, 32, 1, 1}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, ReflectionPad2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/reflection_pad_2d.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Conv2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/conv2d.pth";
  const auto input = torch::randn({2, 3, 32, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 5e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Deconv2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/deconv2d.pth";
  const auto input = torch::randn({1, 8, 32, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 5e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Expand) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/expand.pth";
  const auto input = torch::randn({1, 3, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Pooling2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/pooling_2d.pth";
  const auto x = ::torch::randn({1, 23, 54, 96}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Pooling3d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/pooling_3d.pth";
  const auto x = torch::randn({1, 14, 23, 54, 96}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, AdaptivePooling2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/adaptive_pooling_2d.pth";
  const auto x = torch::randn({1, 64, 10, 9}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, AdaptivePooling3d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/adaptive_pooling_3d.pth";
  const auto x = torch::randn({1, 64, 8, 9, 10}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Cat) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/cat.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Stack) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/stack.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Softmax) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/softmax.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, BatchNorm2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/batch_norm_2d.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, LayerNorm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/layer_norm.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, LayerNormWithWeights) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/layer_norm_with_weights.pth";
  const auto input = torch::randn({20, 5, 10}, device);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float32");
  TestTorchInference(model_path, input_map, "int8_calib", 1e-1,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
  TestTorchInference(model_path, input_map, "int8", 1e-1,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Bmm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/bmm.pth";
  const auto mat1 = ::torch::randn({3, 4, 5}, device).to(c10::kHalf);
  const auto mat2 = ::torch::randn({3, 5, 6}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;

  input_map["x"] = mat1;
  input_map["y"] = mat2;
  TestTorchInference(model_path, input_map, "int8", 1e-1,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Clamp) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/clamp.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, ConstantPad2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/constant_pad_2d.pth";
  const auto input = torch::randn({1, 13, 7, 9}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, ConstantPad3d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/constant_pad_3d.pth";
  const auto input = torch::randn({1, 3, 11, 22, 33}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Floor) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/floor.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Permute) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/permute.pth";
  const auto input = torch::randn({3, 5, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Repeat) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/repeat.pth";
  const auto input = torch::randn({2, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Slice) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/slice.pth";
  const auto input = torch::randn({4, 64, 64, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Split) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/split.pth";
  const auto input = torch::randn({1, 4, 10, 9}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, SplitStack) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/split_stack.pth";
  const auto input = torch::randn({1, 8, 2, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, View) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/view_and_reshape.pth";
  const auto input = torch::randn({5, 4, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, UpsamplingBilinear2dWithSize) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_bilinear_2d_with_size.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, UpsamplingNearest2dWithSize) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_nearest_2d_with_size.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, UpsamplingBilinear2dWithScale) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_bilinear_2d_with_scale.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, UpsamplingNearest2dWithScale) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_nearest_2d_with_scale.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Unsqueeze) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/unsqueeze.pth";
  const auto input = torch::randn({1, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Var) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/var.pth";
  const auto input = torch::randn({3, 13, 41, 39}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, GridSamplerBilinearModule) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/grid_sampler_bilinear.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);
  const auto T = ::torch::randn({1, 5, 10, 2}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["grid"] = T;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, GridSamplerNearestModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/grid_sampler_nearest.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);
  const auto T = ::torch::randn({1, 6, 7, 2}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["grid"] = T;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, LrnModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/lrn.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, ReduceModule) {
  const auto input = torch::randn({32, 16, 45, 12}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce.pth", input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_0.pth", input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_1.pth", input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_2.pth", input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_3.pth", input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

// TEST(TestTorchNodes131Int8, EmbeddingBagModule) {
//   const std::string& model_path =
//       std::string(torch_root_dir) + "nodes131/embedding_bag.pth";
//   auto input = ::torch::tensor({1, 0, 3, 1, 4},
//                                torch::requires_grad(false).dtype(c10::kLong));
//   auto offset = torch::tensor({0, 0, 0, 0, 0},
//                               torch::requires_grad(false).dtype(c10::kLong));
//   std::unordered_map<std::string, c10::IValue> input_map;
//   input_map["input"] = input;
//   input_map["offsets"] = offset;
//
//   TestTorchInference(model_path, input_map, "int8", 1e-2,
//                      std::make_shared<TestBatchStream>(InputVolume(input_map)));
// }

TEST(TestTorchNodes131Int8, IndexModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/index.pth";
  auto input = ::torch::randn({3, 10, 10}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["X"] = input;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, ILN) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/ILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, AdaILN) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/AdaILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device).to(c10::kHalf);
  const auto gamma = ::torch::randn({1, 7}, device).to(c10::kHalf);
  const auto beta = ::torch::randn({1, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["gamma"] = gamma;
  input_map["beta"] = beta;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

#ifdef SUPPORT_RNN
TEST(TestTorchNodes131Int8, RnnModuleTanHBid) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 3e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, BidirectionRNN) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, RnnModuleRelu) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/rnn_relu.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, LstmModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/lstm.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  const auto c_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, Lstm2Module) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/lstm2.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  const auto c_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodes131Int8, GruModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/gru.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}
#endif
