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

#include "unit_test/unit_test_keras_helper.h"

class TestKerasNodes : public ::testing::Test {
 protected:
  void SetUp() override {
    pb_path = std::string(tf_root_dir);
    keras_h5_path = std::string(keras_root_dir);
    // configuration
    infer_mode = "float32";
    threshold = 1e-3;
  };
  void TearDown() override{};
  float threshold{1e-3};
  std::string pb_path;
  std::string keras_h5_path;
  std::string infer_mode;
  std::vector<std::string> output_names;
  std::vector<std::pair<std::string, TF_Tensor*>> input_map;
};

TEST_F(TestKerasNodes, AvgPool) {
  pb_path = pb_path + "average_pooling.pb";
  keras_h5_path = keras_h5_path + "average_pooling.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 29, 17, 3});

  input_map.push_back({"input", input.get()});
  output_names = {"average_pooling2d/AvgPool", "average_pooling2d_1/AvgPool"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Activation) {
  pb_path = pb_path + "activation.pb";
  keras_h5_path = keras_h5_path + "activation.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map.push_back({"input", input.get()});
  output_names = {"activation/Sigmoid", "activation_1/Relu", "activation_2/Tanh",
                  "activation_3/Elu"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Arithmetic) {
  pb_path = pb_path + "arithmetic.pb";
  keras_h5_path = keras_h5_path + "arithmetic.h5";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});
  const auto input2 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map.push_back({"input1", input1.get()});
  input_map.push_back({"input2", input2.get()});
  output_names = {"add/add", "subtract/sub", "multiply/mul"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, BatchNorm) {
  pb_path = pb_path + "batch_norm.pb";
  keras_h5_path = keras_h5_path + "batch_norm.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map.push_back({"input_1", input.get()});
  output_names = {"batch_normalization/FusedBatchNormV3"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Concatenate) {
  pb_path = pb_path + "concatenate.pb";
  keras_h5_path = keras_h5_path + "concatenate.h5";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});
  const auto input2 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});
  const auto input3 = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});

  input_map.push_back({"input1", input1.get()});
  input_map.push_back({"input2", input2.get()});
  input_map.push_back({"input3", input3.get()});
  output_names = {"concatenate/concat", "concatenate_1/concat", "concatenate_2/concat"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Convolution) {
  pb_path = pb_path + "conv2d.pb";
  keras_h5_path = keras_h5_path + "conv2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  input_map.push_back({"input_2", input.get()});
  output_names = {"conv2d/BiasAdd"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Conv2DActivation) {
  pb_path = pb_path + "conv2d_activation.pb";
  keras_h5_path = keras_h5_path + "conv2d_activation.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  input_map.push_back({"input_1", input.get()});
  output_names = {"conv2d/Relu"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Cropping2D) {
  pb_path = pb_path + "cropping2d.pb";
  keras_h5_path = keras_h5_path + "cropping2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  input_map.push_back({"input_1", input.get()});
  output_names = {"cropping2d/strided_slice"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, DepthwiseConv2d) {
  pb_path = pb_path + "depthwise_conv2d.pb";
  keras_h5_path = keras_h5_path + "depthwise_conv2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 11});

  input_map.push_back({"input_1", input.get()});
  output_names = {"depthwise_conv2d/BiasAdd"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Embedding) {
  pb_path = pb_path + "embedding.pb";
  keras_h5_path = keras_h5_path + "embedding.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, 10}, 1000);

  input_map.push_back({"input1_1", input.get()});
  output_names = {"embedding_4/embedding_lookup/Identity_1"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Flatten) {
  pb_path = pb_path + "flatten.pb";
  keras_h5_path = keras_h5_path + "flatten.h5";

  const int batch_size = 1;
  auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map.push_back({"input_6", input.get()});
  output_names = {"flatten/Reshape"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, FullyConnected) {
  pb_path = pb_path + "dense.pb";
  keras_h5_path = keras_h5_path + "dense.h5";

  const int batch_size = 1;
  auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 784});

  input_map.push_back({"input_4", input.get()});
  output_names = {"dense/BiasAdd"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, MaxPool) {
  pb_path = pb_path + "max_pooling.pb";
  keras_h5_path = keras_h5_path + "max_pooling.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 13, 33, 3});

  input_map.push_back({"input", input.get()});
  output_names = {"max_pooling2d/MaxPool", "max_pooling2d_1/MaxPool"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Permute) {
  pb_path = pb_path + "permute.pb";
  keras_h5_path = keras_h5_path + "permute.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map.push_back({"input_8", input.get()});
  output_names = {"permute/transpose"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Reduce) {
  pb_path = pb_path + "reduce.pb";
  keras_h5_path = keras_h5_path + "reduce.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  input_map.push_back({"input1", input.get()});
  output_names = {"tf_op_layer_Mean/Mean", "tf_op_layer_Sum/Sum", "tf_op_layer_Max/Max",
                  "tf_op_layer_Min/Min"},
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, Softmax) {
  pb_path = pb_path + "softmax.pb";
  keras_h5_path = keras_h5_path + "softmax.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map.push_back({"input_11", input.get()});
  output_names = {"softmax/Softmax"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, SeparableConv2d) {
  pb_path = pb_path + "separable_conv2d.pb";
  keras_h5_path = keras_h5_path + "separable_conv2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 11});

  input_map.push_back({"input_1", input.get()});
  output_names = {"separable_conv2d/BiasAdd", "separable_conv2d_1/separable_conv2d"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, ZeroPadding) {
  pb_path = pb_path + "zero_padding_2d.pb";
  keras_h5_path = keras_h5_path + "zero_padding_2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  input_map.push_back({"input_1", input.get()});
  output_names = {"zero_padding2d/Pad", "zero_padding2d_1/Pad", "zero_padding2d_2/Pad",
                  "zero_padding2d_3/Pad"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

#ifdef SUPPORT_RNN
TEST_F(TestKerasNodes, RNNTanh) {
  pb_path = pb_path + "rnn_tanh.pb";
  keras_h5_path = keras_h5_path + "rnn_tanh.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"simple_rnn/strided_slice_3", "simple_rnn_1/transpose_1"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, RNNRelu) {
  pb_path = pb_path + "rnn_relu.pb";
  keras_h5_path = keras_h5_path + "rnn_relu.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"simple_rnn/strided_slice_3", "simple_rnn_1/transpose_1"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, BiRNN) {
  pb_path = pb_path + "bidirectional_rnn.pb";
  keras_h5_path = keras_h5_path + "bidirectional_rnn.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"bidirectional/concat", "bidirectional_1/concat"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, LSTM) {
  pb_path = pb_path + "lstm.pb";
  keras_h5_path = keras_h5_path + "lstm.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"lstm/strided_slice_7", "lstm_1/transpose_1"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, BiLSTM) {
  pb_path = pb_path + "bidirectional_lstm.pb";
  keras_h5_path = keras_h5_path + "bidirectional_lstm.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"bidirectional/concat", "bidirectional_1/concat"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, GRU) {
  pb_path = pb_path + "gru.pb";
  keras_h5_path = keras_h5_path + "gru.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"gru/strided_slice_15", "gru_1/transpose_1"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}

TEST_F(TestKerasNodes, BiGRU) {
  pb_path = pb_path + "bidirectional_gru.pb";
  keras_h5_path = keras_h5_path + "bidirectional_gru.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  input_map.push_back({"input_1", input.get()});
  output_names = {"bidirectional/concat", "bidirectional_1/concat"};
  TestKerasInference(pb_path, keras_h5_path, input_map, output_names, batch_size, threshold);
}
#endif
