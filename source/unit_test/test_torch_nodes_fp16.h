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

TEST(TestTorchNodes131Fp16, Arithmetic) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/arithmetic.pth";
  const auto a = ::torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);
  const auto b = ::torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["a"] = a;
  input_map["b"] = b;

  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Addmm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/addmm.pth";
  const auto M = ::torch::randn({3, 3}, device).to(c10::kHalf);
  const auto mat1 = ::torch::randn({3, 3}, device).to(c10::kHalf);
  const auto mat2 = ::torch::randn({3, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["M"] = M;
  input_map["mat1"] = mat1;
  input_map["mat2"] = mat2;

  // TestTorchInference(model_path, {M, mat1, mat2}, "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, FullyConnected) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/fully_connected.pth";
  const auto input = ::torch::randn({1, 10}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, Activation) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/activation.pth";
  const auto input = ::torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, {input}, "float16");
  // TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, PRelu) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/prelu.pth";
  const auto input = torch::randn({1, 3, 11, 13}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Inplace) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/inplace.pth";
  const auto input = torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, InstanceNorm2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/instance_norm.pth";
  const auto input = torch::randn({3, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, InstanceNorm2dTrack) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/instance_norm_track.pth";
  const auto input = torch::randn({4, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, InstanceNorm2dAffine) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/instance_norm_aff.pth";
  const auto input = torch::randn({5, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, InstanceNorm2dAffineTrack) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/instance_norm_aff_track.pth";
  const auto input = torch::randn({6, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 3e-2);
}

TEST(TestTorchNodes131Fp16, Norm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/norm.pth";
  const auto input = torch::randn({1, 32, 1, 1}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, ReflectionPad2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/reflection_pad_2d.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Conv2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/conv2d.pth";
  const auto input = torch::randn({2, 3, 32, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 3e-2);
}

TEST(TestTorchNodes131Fp16, Deconv2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/deconv2d.pth";
  const auto input = torch::randn({1, 8, 32, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 5e-2);
}

TEST(TestTorchNodes131Fp16, Expand) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/expand.pth";
  const auto input = torch::randn({1, 3, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, MaxPooling) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/max_pooling.pth";
  const auto x1 = ::torch::randn({1, 23, 54, 96}, device).to(c10::kHalf);
  const auto x2 = ::torch::randn({1, 14, 23, 54, 96}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x1;
  input_map["input0"] = x2;
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, AvgPooling) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/avg_pooling.pth";
  const auto x1 = ::torch::randn({1, 23, 54, 96}, device).to(c10::kHalf);
  const auto x2 = ::torch::randn({1, 14, 23, 54, 96}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x1;
  input_map["input0"] = x2;
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, AdaptiveMaxPooling) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/adaptive_max_pooling.pth";
  const auto x1 = ::torch::randn({1, 64, 10, 9}, device).to(c10::kHalf);
  const auto x2 = ::torch::randn({1, 64, 8, 9, 10}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x1;
  input_map["input0"] = x2;
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, AdaptiveAvgPooling) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/adaptive_avg_pooling.pth";
  const auto x1 = ::torch::randn({1, 64, 10, 9}, device).to(c10::kHalf);
  const auto x2 = ::torch::randn({1, 64, 8, 9, 10}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x1;
  input_map["input0"] = x2;
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Cat) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/cat.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Stack) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/stack.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Softmax) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/softmax.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, BatchNorm2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/batch_norm_2d.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, LayerNorm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/layer_norm.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, LayerNormWithWeights) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/layer_norm_with_weights.pth";
  const auto input = torch::randn({20, 5, 10}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float32");
  TestTorchInference(model_path, input_map, "float16", 1e-1);
}

TEST(TestTorchNodes131Fp16, Bmm) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/bmm.pth";
  const auto mat1 = ::torch::randn({3, 4, 5}, device).to(c10::kHalf);
  const auto mat2 = ::torch::randn({3, 5, 6}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;

  input_map["x"] = mat1;
  input_map["y"] = mat2;
  TestTorchInference(model_path, input_map, "float16", 5e-2);
}

TEST(TestTorchNodes131Fp16, Clamp) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/clamp.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, ConstantPad2d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/constant_pad_2d.pth";
  const auto input = torch::randn({1, 13, 7, 9}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, ConstantPad3d) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/constant_pad_3d.pth";
  const auto input = torch::randn({1, 3, 11, 22, 33}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Floor) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/floor.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Permute) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/permute.pth";
  const auto input = torch::randn({3, 5, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Repeat) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/repeat.pth";
  const auto input = torch::randn({2, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Slice) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/slice.pth";
  const auto input = torch::randn({4, 64, 64, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Split) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/split.pth";
  const auto input = torch::randn({1, 4, 10, 9}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, SplitStack) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/split_stack.pth";
  const auto input = torch::randn({1, 8, 2, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, View) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/view_and_reshape.pth";
  const auto input = torch::randn({5, 4, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, UpsamplingBilinear2dWithSize) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_bilinear_2d_with_size.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, UpsamplingNearest2dWithSize) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_nearest_2d_with_size.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, UpsamplingBilinear2dWithScale) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_bilinear_2d_with_scale.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, UpsamplingNearest2dWithScale) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/upsampling_nearest_2d_with_scale.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Unsqueeze) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/unsqueeze.pth";
  const auto input = torch::randn({1, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, Var) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/var.pth";
  const auto input = torch::randn({3, 13, 41, 39}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, GridSamplerBilinearModule) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/grid_sampler_bilinear.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);
  const auto T = ::torch::randn({1, 5, 10, 2}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["grid"] = T;

  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, GridSamplerNearestModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/grid_sampler_nearest.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);
  const auto T = ::torch::randn({1, 6, 7, 2}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["grid"] = T;

  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, LrnModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/lrn.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, ReduceModule) {
  const auto input = torch::randn({32, 16, 45, 12}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_0.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_1.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_2.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes131/reduce_3.pth", input_map, "float16");
}

TEST(TestTorchNodes131Fp16, EmbeddingBagModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/embedding_bag.pth";
  auto input = ::torch::tensor({1, 0, 3, 1, 4}, torch::requires_grad(false).dtype(c10::kLong));
  auto offset = torch::tensor({0, 0, 0, 0, 0}, torch::requires_grad(false).dtype(c10::kLong));
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["offsets"] = offset;

  // TestTorchInference(model_path, std::vector<c10::IValue>{input, offset},
  // "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, IndexModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/index.pth";
  auto input = ::torch::randn({3, 10, 10}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["X"] = input;
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, ILN) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/ILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, AdaILN) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/AdaILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device).to(c10::kHalf);
  const auto gamma = ::torch::randn({1, 7}, device).to(c10::kHalf);
  const auto beta = ::torch::randn({1, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["gamma"] = gamma;
  input_map["beta"] = beta;

  // TestTorchInference(model_path, {input, gamma, beta}, "float16");
  TestTorchInference(model_path, input_map, "float16", 3e-2);
}

#ifdef SUPPORT_RNN
TEST(TestTorchNodes131Fp16, RnnModuleTanHBid) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, BidirectionRNN) {
  const std::string& model_path =
      std::string(torch_root_dir) + "nodes131/rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, RnnModuleRelu) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/rnn_relu.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, LstmModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/lstm.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  // 模型需要在Torch里跑一遍，所以此处batch_size在第二维
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  const auto c_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16");
}

TEST(TestTorchNodes131Fp16, Lstm2Module) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/lstm2.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  // 模型需要在Torch里跑一遍，所以此处batch_size在第二维
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  const auto c_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "float16", 1e-2);
}

TEST(TestTorchNodes131Fp16, GruModule) {
  const std::string& model_path = std::string(torch_root_dir) + "nodes131/gru.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;

  TestTorchInference(model_path, input_map, "float16");
}
#endif  // SUPPORT_RNN

// TEST(TestTorchNodes, MultiThread) {
//   const std::string engine_path = std::string(torch_root_dir) +
//   "nodes/temp.engine"; TorchInfer torch_infer; TorchEngine torch_engine; auto
//   input = ::torch::randn({1, 3, 7, 7});
//   TestTorchLoadModule(std::string(torch_root_dir) + "nodes/activation.pth",
//   engine_path,
//                       {input}, torch_infer, torch_engine, "float16");
//   for (int t = 0; t < 100; ++t) {
//     std::vector<std::thread> t_vec;
//     for (int i = 0; i < 10; ++i) {
//       std::thread t(
//           [](const std::string& engine_path) {
//             TorchEngine engine;
//             if (!engine.Load(engine_path)) {
//               std::cout << "Load Engine failed." << std::endl;
//               return;
//             }
//             auto input = ::torch::randn({1, 3, 7, 7}, c10::kCPU);
//             // Forward
//             for (int i = 0; i < 100; ++i) {
//               engine.Forward(input);
//             }
//           },
//           engine_path);
//       t_vec.push_back(std::move(t));
//     }
//     for (auto& t : t_vec) {
//       t.join();
//     }
//   }
// }
