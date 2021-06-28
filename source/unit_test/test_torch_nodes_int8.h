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

TEST(TestTorchNodesInt8, AdaILN) {
  const auto model_path = std::string(torch_root_dir) + "nodes/AdaILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device).to(c10::kHalf);
  const auto gamma = ::torch::randn({1, 7}, device).to(c10::kHalf);
  const auto beta = ::torch::randn({1, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["gamma"] = gamma;
  input_map["beta"] = beta;

  // TestTorchInference(model_path, {input, gamma, beta}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Arithmetic) {
  const auto model_path = std::string(torch_root_dir) + "nodes/arithmetic.pth";
  const auto a = ::torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);
  const auto b = ::torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["a"] = a;
  input_map["b"] = b;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Addmm) {
  const auto model_path = std::string(torch_root_dir) + "nodes/addmm.pth";
  const auto M = ::torch::randn({3, 3}, device).to(c10::kHalf);
  const auto mat1 = ::torch::randn({3, 3}, device).to(c10::kHalf);
  const auto mat2 = ::torch::randn({3, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["M"] = M;
  input_map["mat1"] = mat1;
  input_map["mat2"] = mat2;

  // TestTorchInference(model_path, {M, mat1, mat2}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Activation) {
  const auto model_path = std::string(torch_root_dir) + "nodes/activation.pth";
  const auto input = ::torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, AdaptivePooling2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/adaptive_pooling_2d.pth";
  const auto x = torch::randn({1, 64, 10, 9}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, AdaptivePooling3d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/adaptive_pooling_3d.pth";
  const auto x = torch::randn({1, 64, 8, 9, 10}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, BatchNorm2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/batch_norm_2d.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Bmm) {
  const auto model_path = std::string(torch_root_dir) + "nodes/bmm.pth";
  const auto mat1 = ::torch::randn({3, 4, 5}, device).to(c10::kHalf);
  const auto mat2 = ::torch::randn({3, 5, 6}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;

  input_map["x"] = mat1;
  input_map["y"] = mat2;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Clamp) {
  const auto model_path = std::string(torch_root_dir) + "nodes/clamp.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Cat) {
  const auto model_path = std::string(torch_root_dir) + "nodes/cat.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Conv2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/conv2d.pth";
  const auto input = torch::randn({2, 3, 32, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, ConstantPad2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/constant_pad_2d.pth";
  const auto input = torch::randn({1, 13, 7, 9}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, ConstantPad3d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/constant_pad_3d.pth";
  const auto input = torch::randn({1, 3, 11, 22, 33}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Deconv2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/deconv2d.pth";
  const auto input = torch::randn({1, 8, 32, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Expand) {
  const auto model_path = std::string(torch_root_dir) + "nodes/expand.pth";
  const auto input = torch::randn({1, 3, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, EmbeddingBagModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/embedding_bag.pth";
  auto input = ::torch::tensor({1, 0, 3, 1, 4}, torch::requires_grad(false).dtype(c10::kLong));
  auto offset = torch::tensor({0, 0, 0, 0, 0}, torch::requires_grad(false).dtype(c10::kLong));
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["offsets"] = offset;

  // TestTorchInference(model_path, std::vector<c10::IValue>{input, offset},
  // "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, FullyConnected) {
  const auto model_path = std::string(torch_root_dir) + "nodes/fully_connected.pth";
  const auto input = ::torch::randn({1, 10}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, {input}, "float16");
  // TestTorchInference(model_path, input_map, "int8", 1e-2,
  // std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Floor) {
  const auto model_path = std::string(torch_root_dir) + "nodes/floor.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Gelu) {
  const auto model_path = std::string(torch_root_dir) + "nodes/gelu.pth";
  const auto input = torch::randn({1, 3, 24, 24}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, GridSamplerBilinearModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/grid_sampler_bilinear.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);
  const auto T = ::torch::randn({1, 5, 10, 2}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["grid"] = T;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, GridSamplerNearestModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/grid_sampler_nearest.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);
  const auto T = ::torch::randn({1, 6, 7, 2}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["grid"] = T;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, IndexModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/index.pth";
  auto input = ::torch::randn({3, 10, 10});
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["X"] = input;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, ILN) {
  const auto model_path = std::string(torch_root_dir) + "nodes/ILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Inplace) {
  const auto model_path = std::string(torch_root_dir) + "nodes/inplace.pth";
  const auto input = torch::randn({1, 3, 64, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, InstanceNorm2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/instance_norm.pth";
  const auto input = torch::randn({3, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, InstanceNorm2dTrack) {
  const auto model_path = std::string(torch_root_dir) + "nodes/instance_norm_track.pth";
  const auto input = torch::randn({4, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, InstanceNorm2dAffine) {
  const auto model_path = std::string(torch_root_dir) + "nodes/instance_norm_aff.pth";
  const auto input = torch::randn({5, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, InstanceNorm2dAffineTrack) {
  const auto model_path = std::string(torch_root_dir) + "nodes/instance_norm_aff_track.pth";
  const auto input = torch::randn({6, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, LayerNorm) {
  const auto model_path = std::string(torch_root_dir) + "nodes/layer_norm.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, LayerNormWithWeights) {
  const auto model_path = std::string(torch_root_dir) + "nodes/layer_norm_with_weights.pth";
  const auto input = torch::randn({20, 5, 10}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, LrnModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/lrn.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Norm) {
  const auto model_path = std::string(torch_root_dir) + "nodes/norm.pth";
  const auto input = torch::randn({1, 32, 1, 1}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, PRelu) {
  const auto model_path = std::string(torch_root_dir) + "nodes/prelu.pth";
  const auto input = torch::randn({1, 3, 11, 13}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Pooling2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/pooling_2d.pth";
  const auto x = ::torch::randn({1, 23, 54, 96}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Pooling3d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/pooling_3d.pth";
  const auto x = torch::randn({1, 14, 23, 54, 96}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Permute) {
  const auto model_path = std::string(torch_root_dir) + "nodes/permute.pth";
  const auto input = torch::randn({3, 5, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

#ifdef ENABLE_TORCH_PLUGIN
TEST(TestTorchNodesInt8, PixelShuffle) {
  const auto model_path = std::string(torch_root_dir) + "nodes/pixel_shuffle.pth";
  const auto input = torch::randn({1, 9, 24, 24}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}
#endif

TEST(TestTorchNodesInt8, ReduceModule) {
  const auto input = torch::randn({32, 16, 45, 12}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  TestTorchInference(std::string(torch_root_dir) + "nodes/reduce.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes/reduce_0.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes/reduce_1.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes/reduce_2.pth", input_map, "float16");
  TestTorchInference(std::string(torch_root_dir) + "nodes/reduce_3.pth", input_map, "float16");
}

TEST(TestTorchNodesInt8, ReflectionPad2d) {
  const auto model_path = std::string(torch_root_dir) + "nodes/reflection_pad_2d.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Repeat) {
  const auto model_path = std::string(torch_root_dir) + "nodes/repeat.pth";
  const auto input = torch::randn({2, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Stack) {
  const auto model_path = std::string(torch_root_dir) + "nodes/stack.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Softmax) {
  const auto model_path = std::string(torch_root_dir) + "nodes/softmax.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Slice) {
  const auto model_path = std::string(torch_root_dir) + "nodes/slice.pth";
  const auto input = torch::randn({4, 64, 64, 64}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Split) {
  const auto model_path = std::string(torch_root_dir) + "nodes/split.pth";
  const auto input = torch::randn({1, 4, 10, 9}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, SplitStack) {
  const auto model_path = std::string(torch_root_dir) + "nodes/split_stack.pth";
  const auto input = torch::randn({1, 8, 2, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["argument_1"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, UpsamplingBilinear2dWithSize) {
  const auto model_path =
      std::string(torch_root_dir) + "nodes/upsampling_bilinear_2d_with_size.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, UpsamplingNearest2dWithSize) {
  const auto model_path = std::string(torch_root_dir) + "nodes/upsampling_nearest_2d_with_size.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, UpsamplingBilinear2dWithScale) {
  const auto model_path =
      std::string(torch_root_dir) + "nodes/upsampling_bilinear_2d_with_scale.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, UpsamplingNearest2dWithScale) {
  const auto model_path =
      std::string(torch_root_dir) + "nodes/upsampling_nearest_2d_with_scale.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Unsqueeze) {
  const auto model_path = std::string(torch_root_dir) + "nodes/unsqueeze.pth";
  const auto input = torch::randn({1, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Var) {
  const auto model_path = std::string(torch_root_dir) + "nodes/var.pth";
  const auto input = torch::randn({3, 13, 41, 39}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, View) {
  const auto model_path = std::string(torch_root_dir) + "nodes/view_and_reshape.pth";
  const auto input = torch::randn({5, 4, 3}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["x"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

#ifdef SUPPORT_RNN
TEST(TestTorchNodesInt8, BidirectionRNN) {
  const auto model_path = std::string(torch_root_dir) + "nodes/rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, LstmModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/lstm.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  // 模型需要在Torch里跑一遍，所以此处batch_size在第二维
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  const auto c_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, Lstm2Module) {
  const auto model_path = std::string(torch_root_dir) + "nodes/lstm2.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  // 模型需要在Torch里跑一遍，所以此处batch_size在第二维
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  const auto c_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, GruModule) {
  const auto model_path = std::string(torch_root_dir) + "nodes/gru.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);
  const auto h_0 = ::torch::randn({1, 2, 128}, device).to(c10::kHalf);
  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;
  input_map["h_0"] = h_0;

  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, RnnModuleTanHBid) {
  const auto model_path = std::string(torch_root_dir) + "nodes/rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

TEST(TestTorchNodesInt8, RnnModuleRelu) {
  const auto model_path = std::string(torch_root_dir) + "nodes/rnn_relu.pth";
  const auto input = torch::randn({1, 28, 28}, device).to(c10::kHalf);

  std::unordered_map<std::string, c10::IValue> input_map;
  input_map["input"] = input;

  // TestTorchInference(model_path, {input}, "float16");
  TestTorchInference(model_path, input_map, "int8", 1e-2,
                     std::make_shared<TestBatchStream>(InputVolume(input_map)));
}

#endif  // SUPPORT_RNN
