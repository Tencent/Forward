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

#include "unit_test/unit_test_torch_helper.h"

class TestTorchNodes : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path = std::string(torch_root_dir) + "nodes/";
    // configuration
    infer_mode = "float32";
    threshold = 1e-3;
  };
  void TearDown() override{};
  std::string model_path;
  std::string infer_mode;
  float threshold{1e-3};
  std::unordered_map<std::string, c10::IValue> input_map;
};

TEST_F(TestTorchNodes, AdaILN) {
  model_path = model_path + "AdaILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device);
  const auto gamma = ::torch::randn({1, 7}, device);
  const auto beta = ::torch::randn({1, 7}, device);

  input_map["input"] = input;
  input_map["gamma"] = gamma;
  input_map["beta"] = beta;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Arithmetic) {
  model_path = model_path + "arithmetic.pth";
  const auto a = ::torch::randn({1, 3, 64, 32}, device);
  const auto b = ::torch::randn({1, 3, 64, 32}, device);
  input_map["a"] = a;
  input_map["b"] = b;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Addmm) {
  model_path = model_path + "addmm.pth";
  const auto M = ::torch::randn({3, 3}, device);
  const auto mat1 = ::torch::randn({3, 3}, device);
  const auto mat2 = ::torch::randn({3, 3}, device);

  input_map["M"] = M;
  input_map["mat1"] = mat1;
  input_map["mat2"] = mat2;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Activation) {
  model_path = model_path + "activation.pth";
  const auto input = ::torch::randn({1, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, AdaptivePooling2d) {
  model_path = model_path + "adaptive_pooling_2d.pth";
  const auto x = torch::randn({1, 64, 10, 9}, device);
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, AdaptivePooling3d) {
  model_path = model_path + "adaptive_pooling_3d.pth";
  const auto x = torch::randn({1, 64, 8, 9, 10}, device);
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, BatchNorm2d) {
  model_path = model_path + "batch_norm_2d.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Bmm) {
  model_path = model_path + "bmm.pth";
  const auto mat1 = ::torch::randn({3, 4, 5}, device);
  const auto mat2 = ::torch::randn({3, 5, 6}, device);

  input_map["x"] = mat1;
  input_map["y"] = mat2;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Clamp) {
  model_path = model_path + "clamp.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Cat) {
  model_path = model_path + "cat.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Conv2d) {
  model_path = model_path + "conv2d.pth";
  const auto input = torch::randn({2, 3, 32, 64}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, ConstantPad2d) {
  model_path = model_path + "constant_pad_2d.pth";
  const auto input = torch::randn({1, 13, 7, 9}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, ConstantPad3d) {
  model_path = model_path + "constant_pad_3d.pth";
  const auto input = torch::randn({1, 3, 11, 22, 33}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Deconv2d) {
  model_path = model_path + "deconv2d.pth";
  const auto input = torch::randn({1, 8, 32, 64}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Expand) {
  model_path = model_path + "expand.pth";
  const auto input = torch::randn({1, 3, 2}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, EmbeddingBagModule) {
  model_path = model_path + "embedding_bag.pth";
  auto input = ::torch::tensor({1, 0, 3, 1, 4}, torch::requires_grad(false).dtype(c10::kLong));
  auto offset = torch::tensor({0, 0, 0, 0, 0}, torch::requires_grad(false).dtype(c10::kLong));
  input_map["input"] = input;
  input_map["offsets"] = offset;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, FullyConnected) {
  model_path = model_path + "fully_connected.pth";
  const auto input = ::torch::randn({1, 10}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Floor) {
  model_path = model_path + "floor.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Gelu) {
  model_path = model_path + "gelu.pth";
  const auto input = torch::randn({1, 3, 24, 24}, device);

  input_map["input"] = input;

  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, GridSamplerBilinearModule) {
  model_path = model_path + "grid_sampler_bilinear.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device);
  const auto T = ::torch::randn({1, 5, 10, 2}, device);
  input_map["input"] = input;
  input_map["grid"] = T;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, GridSamplerNearestModule) {
  model_path = model_path + "grid_sampler_nearest.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device);
  const auto T = ::torch::randn({1, 6, 7, 2}, device);
  input_map["input"] = input;
  input_map["grid"] = T;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, IndexModule) {
  model_path = model_path + "index.pth";
  auto input = ::torch::randn({3, 10, 10});
  input_map["X"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, ILN) {
  model_path = model_path + "ILN.pth";
  const auto input = ::torch::randn({1, 7, 231, 343}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Inplace) {
  model_path = model_path + "inplace.pth";
  const auto input = torch::randn({1, 3, 64, 32}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, InstanceNorm2d) {
  model_path = model_path + "instance_norm.pth";
  const auto input = torch::randn({3, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, InstanceNorm2dTrack) {
  model_path = model_path + "instance_norm_track.pth";
  const auto input = torch::randn({4, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, InstanceNorm2dAffine) {
  model_path = model_path + "instance_norm_aff.pth";
  const auto input = torch::randn({5, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, InstanceNorm2dAffineTrack) {
  model_path = model_path + "instance_norm_aff_track.pth";
  const auto input = torch::randn({6, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, LayerNorm) {
  model_path = model_path + "layer_norm.pth";
  const auto input = torch::randn({3, 3, 14, 32}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, LayerNormWithWeights) {
  model_path = model_path + "layer_norm_with_weights.pth";
  const auto input = torch::randn({20, 5, 10}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, LrnModule) {
  model_path = model_path + "lrn.pth";
  const auto input = torch::randn({1, 3, 5, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, MatMul) {
  model_path = model_path + "matmul.pth";
  const auto mat1 = ::torch::randn({3, 4, 5}, device);
  const auto mat2 = ::torch::randn({3, 5, 6}, device);

  input_map["x"] = mat1;
  input_map["y"] = mat2;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Norm) {
  model_path = model_path + "norm.pth";
  const auto input = torch::randn({1, 32, 1, 1}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, PRelu) {
  model_path = model_path + "prelu.pth";
  const auto input = torch::randn({1, 3, 11, 13}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Pooling2d) {
  model_path = model_path + "pooling_2d.pth";
  const auto x = ::torch::randn({1, 23, 54, 96}, device);
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Pooling3d) {
  model_path = model_path + "pooling_3d.pth";
  const auto x = torch::randn({1, 14, 23, 54, 96}, device);
  input_map["input"] = x;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Permute) {
  model_path = model_path + "permute.pth";
  const auto input = torch::randn({3, 5, 7}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, PixelShuffle) {
  model_path = model_path + "pixel_shuffle.pth";
  const auto input = torch::randn({1, 9, 24, 24}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, ReduceModule) {
  const auto input = torch::randn({32, 16, 45, 12}, device);
  input_map["x"] = input;

  TestTorchInference(model_path + "reduce.pth", input_map, infer_mode, threshold);
  TestTorchInference(model_path + "reduce_0.pth", input_map, infer_mode, threshold);
  TestTorchInference(model_path + "reduce_1.pth", input_map, infer_mode, threshold);
  TestTorchInference(model_path + "reduce_2.pth", input_map, infer_mode, threshold);
  TestTorchInference(model_path + "reduce_3.pth", input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, ReflectionPad2d) {
  model_path = model_path + "reflection_pad_2d.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Repeat) {
  model_path = model_path + "repeat.pth";
  const auto input = torch::randn({2, 3}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Stack) {
  model_path = model_path + "stack.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Softmax) {
  model_path = model_path + "softmax.pth";
  const auto input = torch::randn({1, 3, 7, 7}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Slice) {
  model_path = model_path + "slice.pth";
  const auto input = torch::randn({4, 64, 64, 64}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Split) {
  model_path = model_path + "split.pth";
  const auto input = torch::randn({1, 4, 10, 9}, device);

  input_map["argument_1"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, SplitStack) {
  model_path = model_path + "split_stack.pth";
  const auto input = torch::randn({1, 8, 2, 3}, device);

  input_map["argument_1"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, UpsamplingBilinear2dWithSize) {
  model_path = model_path + "upsampling_bilinear_2d_with_size.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, UpsamplingNearest2dWithSize) {
  model_path = model_path + "upsampling_nearest_2d_with_size.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, UpsamplingBilinear2dWithScale) {
  model_path = model_path + "upsampling_bilinear_2d_with_scale.pth";
  const auto input = torch::randn({1, 128, 20, 20}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, UpsamplingNearest2dWithScale) {
  model_path = model_path + "upsampling_nearest_2d_with_scale.pth";
  const auto input = torch::randn({1, 1, 2, 2}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Unsqueeze) {
  model_path = model_path + "unsqueeze.pth";
  const auto input = torch::randn({1, 3}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Var) {
  model_path = model_path + "var.pth";
  const auto input = torch::randn({3, 13, 41, 39}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, View) {
  model_path = model_path + "view_and_reshape.pth";
  const auto input = torch::randn({5, 4, 3}, device);

  input_map["x"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

#ifdef SUPPORT_RNN
TEST_F(TestTorchNodes, BidirectionRNN) {
  model_path = model_path + "rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, LstmModule) {
  model_path = model_path + "lstm.pth";
  const auto input = torch::randn({1, 28, 28}, device);
  // 模型需要在Torch里跑一遍，所以此处batch_size在第二维
  const auto h_0 = ::torch::randn({1, 2, 128}, device);
  const auto c_0 = ::torch::randn({1, 2, 128}, device);

  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, Lstm2Module) {
  model_path = model_path + "lstm2.pth";
  const auto input = torch::randn({1, 28, 28}, device);
  // 模型需要在Torch里跑一遍，所以此处batch_size在第二维
  const auto h_0 = ::torch::randn({1, 2, 128}, device);
  const auto c_0 = ::torch::randn({1, 2, 128}, device);
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  input_map["c_0"] = c_0;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, GruModule) {
  model_path = model_path + "gru.pth";
  const auto input = torch::randn({1, 28, 28}, device);
  const auto h_0 = ::torch::randn({1, 2, 128}, device);
  input_map["input"] = input;
  input_map["h_0"] = h_0;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, RnnModuleTanHBid) {
  model_path = model_path + "rnn_tanh_bidirectional.pth";
  const auto input = torch::randn({1, 28, 28}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}

TEST_F(TestTorchNodes, RnnModuleRelu) {
  model_path = model_path + "rnn_relu.pth";
  const auto input = torch::randn({1, 28, 28}, device);

  input_map["input"] = input;
  TestTorchInference(model_path, input_map, infer_mode, threshold);
}
#endif  // SUPPORT_RNN
