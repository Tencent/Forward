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

#include <tensorflow/c/c_api.h>  // TensorFlow C API header.

#include <string>

#include "unit_test/unit_test_tf_helper.h"

class TestTfNodes : public ::testing::Test {
 protected:
  void SetUp() override {
    filename = std::string(tf_root_dir);
    // configuration
    infer_mode = "float32";
    threshold = 1e-3;
  };
  void TearDown() override{};
  float threshold{1e-3};
  std::string filename;
  std::string infer_mode;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, TF_Tensor*> input_map;
};

TEST_F(TestTfNodes, HalfConvertors) {
  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomIntTensor<uint16_t>(TF_HALF, {batch_size, 12, 12, 3});
  const auto output = fwd::tf_::CastTensor<float>(input.get(), TF_FLOAT);

  auto gt = fwd::tf_::GetTensorData<uint16_t>(input.get());
  auto res = fwd::tf_::GetTensorData<float>(output.get());
  EXPECT_EQ(gt.size(), res.size());
  for (int i = 0; i < res.size(); ++i) {
    EXPECT_EQ(fwd::FwdUtils::Half2FloatFast(gt[i]), res[i]);
    EXPECT_EQ(gt[i], fwd::FwdUtils::Float2Half(res[i]));
  }
}

TEST_F(TestTfNodes, Activation) {
  filename = filename + "activation.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map["input"] = input.get();
  output_names = {"activation/Sigmoid", "activation_1/Relu", "activation_2/Tanh",
                  "activation_3/Elu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Arithmetic) {
  filename = filename + "arithmetic.pb";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});
  const auto input2 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map["input1"] = input1.get();
  input_map["input2"] = input2.get();
  output_names = {"add/add", "subtract/sub", "multiply/mul"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, AvgPool) {
  filename = filename + "average_pooling.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 29, 17, 3});

  input_map["input"] = input.get();
  output_names = {"average_pooling2d/AvgPool", "average_pooling2d_1/AvgPool"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, BatchNorm) {
  filename = filename + "batch_norm.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map["input_1"] = input.get();
  output_names = {"batch_normalization/FusedBatchNormV3"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Concatenate) {
  filename = filename + "concatenate.pb";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});
  const auto input2 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});
  const auto input3 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});

  input_map["input1"] = input1.get();
  input_map["input2"] = input2.get();
  input_map["input3"] = input3.get();
  output_names = {"concatenate/concat", "concatenate_1/concat", "concatenate_2/concat"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Convolution1d) {
  filename = filename + "conv1d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29});

  input_map["input_9"] = input.get();
  output_names = {"conv1d_6/BiasAdd"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Convolution2d) {
  filename = filename + "conv2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  input_map["input_2"] = input.get();
  output_names = {"conv2d/BiasAdd"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, DepthToSpace) {
  filename = filename + "depth_to_space.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 16});

  input_map["x"] = input.get();
  output_names = {"DepthToSpace"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, DepthwiseConv2d) {
  filename = filename + "depthwise_conv2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 11});

  input_map["input_1"] = input.get();
  output_names = {"depthwise_conv2d/BiasAdd"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Embedding) {
  filename = filename + "embedding.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, 10}, 1000);

  input_map["input1_1"] = input.get();
  output_names = {"embedding_4/embedding_lookup/Identity_1"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, EmbeddingBag) {
  filename = filename + "embedding_bag.pb";

  const int batch_size = 10;
  const int emb_size = 5;

  const auto indices = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, emb_size}, 50);

  input_map["Placeholder"] = indices.get();
  output_names = {"sub"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, FullyConnected) {
  filename = filename + "dense.pb";

  auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {1, 784});

  input_map["input_4"] = input.get();
  output_names = {"dense/BiasAdd"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, MaxPool) {
  filename = filename + "max_pooling.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 13, 33, 3});

  input_map["input"] = input.get();
  output_names = {"max_pooling2d/MaxPool", "max_pooling2d_1/MaxPool"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Permute) {
  filename = filename + "permute.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map["input_8"] = input.get();
  output_names = {"permute/transpose"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Reduce) {
  filename = filename + "reduce.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map["input1"] = input.get();
  output_names = {"tf_op_layer_Mean/Mean", "tf_op_layer_Sum/Sum", "tf_op_layer_Max/Max",
                  "tf_op_layer_Min/Min"};

  TestTFInference(filename, infer_mode, input_map, output_names, 1e-3);
}

TEST_F(TestTfNodes, SeparableConv2d) {
  filename = filename + "separable_conv2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 11});

  input_map["input_1"] = input.get();
  output_names = {"separable_conv2d/BiasAdd", "separable_conv2d_1/separable_conv2d"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Reshape) {
  filename = filename + "reshape.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map["input_9"] = input.get();
  output_names = {"reshape/Reshape"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, SliceStride) {
  filename = filename + "cropping2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  input_map["input_1"] = input.get();
  output_names = {"cropping2d/strided_slice"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Softmax) {
  filename = filename + "softmax.pb";

  const int batch_size = 16;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map["input_11"] = input.get();
  output_names = {"softmax/Softmax"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, Split) {
  filename = filename + "split.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 13, 20});

  input_map["input_1"] = input.get();
  output_names = {"activation/Relu", "activation_1/Relu", "activation_2/Relu", "activation_3/Relu"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}

TEST_F(TestTfNodes, ZeroPadding) {
  filename = filename + "zero_padding_2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map["input_1"] = input.get();
  output_names = {"zero_padding2d/Pad", "zero_padding2d_1/Pad", "zero_padding2d_2/Pad",
                  "zero_padding2d_3/Pad"};

  TestTFInference(filename, infer_mode, input_map, output_names, threshold);
}
