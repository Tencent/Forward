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

#include "unit_test/unit_test.h"

TEST(TestTfNodesInt8, Softmax) {
  const std::string filename = std::string(tf_root_dir) + "softmax.pb";

  const int batch_size = 16;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 12, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_11"] = input.get();
  const std::vector<std::string> output_names{"softmax/Softmax"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Activation) {
  const std::string filename = std::string(tf_root_dir) + "activation.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 24, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input"] = input.get();
  const std::vector<std::string> output_names{"activation/Sigmoid", "activation_1/Relu",
                                              "activation_2/Tanh", "activation_3/Elu"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, BatchNorm) {
  const std::string filename = std::string(tf_root_dir) + "batch_norm.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 24, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"batch_normalization/FusedBatchNormV3"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Arithmetic) {
  const std::string filename = std::string(tf_root_dir) + "arithmetic.pb";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 24, 24, 3});
  const auto input2 = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 24, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input1"] = input1.get();
  input_map["input2"] = input2.get();
  const std::vector<std::string> output_names{"add/add", "subtract/sub", "multiply/mul"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(
                      InputVolume(std::vector<std::shared_ptr<TF_Tensor>>{input1, input2})));
}

TEST(TestTfNodesInt8, Concatenate) {
  const std::string filename = std::string(tf_root_dir) + "concatenate.pb";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 25, 3});
  const auto input2 = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 25, 3});
  const auto input3 = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 25, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input1"] = input1.get();
  input_map["input2"] = input2.get();
  input_map["input3"] = input3.get();
  const std::vector<std::string> output_names{"concatenate/concat", "concatenate_1/concat",
                                              "concatenate_2/concat"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input1, input2, input3})));
}

TEST(TestTfNodesInt8, Convolution1d) {
  const std::string filename = std::string(tf_root_dir) + "conv1d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 29});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_9"] = input.get();
  const std::vector<std::string> output_names{"conv1d_6/BiasAdd"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Convolution2d) {
  const std::string filename = std::string(tf_root_dir) + "conv2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 29, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_2"] = input.get();
  const std::vector<std::string> output_names{"conv2d/BiasAdd"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Split) {
  const std::string filename = std::string(tf_root_dir) + "split.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 13, 20});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"activation/Relu", "activation_1/Relu",
                                              "activation_2/Relu", "activation_3/Relu"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, SeparableConv2d) {
  const std::string filename = std::string(tf_root_dir) + "separable_conv2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 29, 11});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"separable_conv2d/BiasAdd",
                                              "separable_conv2d_1/separable_conv2d"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, DepthwiseConv2d) {
  const std::string filename = std::string(tf_root_dir) + "depthwise_conv2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 29, 11});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"depthwise_conv2d/BiasAdd"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, MaxPool) {
  const std::string filename = std::string(tf_root_dir) + "max_pooling.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 13, 33, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input"] = input.get();
  const std::vector<std::string> output_names{"max_pooling2d/MaxPool", "max_pooling2d_1/MaxPool"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, AvgPool) {
  const std::string filename = std::string(tf_root_dir) + "average_pooling.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 29, 17, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input"] = input.get();
  const std::vector<std::string> output_names{"average_pooling2d/AvgPool",
                                              "average_pooling2d_1/AvgPool"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, FullyConnected) {
  const std::string filename = std::string(tf_root_dir) + "dense.pb";

  auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {1, 784});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_4"] = input.get();
  const std::vector<std::string> output_names{"dense/BiasAdd"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Permute) {
  const std::string filename = std::string(tf_root_dir) + "permute.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 12, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_8"] = input.get();
  const std::vector<std::string> output_names{"permute/transpose"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Reduce) {
  const std::string filename = std::string(tf_root_dir) + "reduce.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 24, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input1"] = input.get();
  const std::vector<std::string> output_names{"tf_op_layer_Mean/Mean", "tf_op_layer_Sum/Sum",
                                              "tf_op_layer_Max/Max", "tf_op_layer_Min/Min"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, SliceStride) {
  const std::string filename = std::string(tf_root_dir) + "cropping2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 23, 29, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"cropping2d/strided_slice"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, ZeroPadding) {
  const std::string filename = std::string(tf_root_dir) + "zero_padding_2d.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<uint16_t>(TF_HALF, {batch_size, 12, 24, 3});

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input_1"] = input.get();
  const std::vector<std::string> output_names{"zero_padding2d/Pad", "zero_padding2d_1/Pad",
                                              "zero_padding2d_2/Pad", "zero_padding2d_3/Pad"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, Embedding) {
  const std::string filename = std::string(tf_root_dir) + "embedding.pb";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, 10}, 1000);

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["input1_1"] = input.get();
  const std::vector<std::string> output_names{"embedding_4/embedding_lookup/Identity_1"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}

TEST(TestTfNodesInt8, EmbeddingBag) {
  const std::string filename = std::string(tf_root_dir) + "embedding_bag.pb";

  const int batch_size = 10;
  const int emb_size = 5;

  const auto input = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, emb_size}, 50);

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["Placeholder"] = input.get();
  const std::vector<std::string> output_names{"sub"};

  TestTFInference(filename, "int8", input_map, output_names, 1e-3,
                  std::make_shared<TestBatchStream>(InputVolume({input})));
}
