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

#include "unit_test/unit_test.h"

TEST(TestKerasNodes, Version) { std::cout << "TensorFlow Version: " << TF_Version() << std::endl; }

TEST(TestKerasNodes, Softmax) {
  const std::string pb_path = std::string(tf_root_dir) + "softmax.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "softmax.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_11"}, {"softmax/Softmax"},
                     batch_size);
}

TEST(TestKerasNodes, Activation) {
  const std::string pb_path = std::string(tf_root_dir) + "activation.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "activation.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  TestKerasInference(
      pb_path, keras_h5_path, {input.get()}, {"input"},
      {"activation/Sigmoid", "activation_1/Relu", "activation_2/Tanh", "activation_3/Elu"},
      batch_size);
}

TEST(TestKerasNodes, BatchNorm) {
  const std::string pb_path = std::string(tf_root_dir) + "batch_norm.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "batch_norm.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"batch_normalization/FusedBatchNormV3"}, batch_size);
}

TEST(TestKerasNodes, Arithmetic) {
  const std::string pb_path = std::string(tf_root_dir) + "arithmetic.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "arithmetic.h5";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});
  const auto input2 = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  TestKerasInference(pb_path, keras_h5_path, {input1.get(), input2.get()}, {"input1", "input2"},
                     {"add/add", "subtract/sub", "multiply/mul"}, batch_size);
}

TEST(TestKerasNodes, Concatenate) {
  const std::string pb_path = std::string(tf_root_dir) + "concatenate.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "concatenate.h5";

  const int batch_size = 1;
  const auto input1 = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});
  const auto input2 = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});
  const auto input3 = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 25, 3});

  TestKerasInference(pb_path, keras_h5_path, {input1.get(), input2.get(), input3.get()},
                     {"input1", "input2", "input3"},
                     {"concatenate/concat", "concatenate_1/concat", "concatenate_2/concat",
                      "concatenate_3/concat"},
                     batch_size);
}

TEST(TestKerasNodes, Convolution) {
  const std::string pb_path = std::string(tf_root_dir) + "conv2d.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "conv2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_2"}, {"conv2d/BiasAdd"},
                     batch_size);
}

TEST(TestKerasNodes, Cropping2D) {
  const std::string pb_path = std::string(tf_root_dir) + "cropping2d.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "cropping2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"cropping2d/strided_slice"}, batch_size);
}

TEST(TestKerasNodes, DepthwiseConv2d) {
  const std::string pb_path = std::string(tf_root_dir) + "depthwise_conv2d.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "depthwise_conv2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 11});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"depthwise_conv2d/BiasAdd"}, batch_size);
}

TEST(TestKerasNodes, SeparableConv2d) {
  const std::string pb_path = std::string(tf_root_dir) + "separable_conv2d.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "separable_conv2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 11});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"separable_conv2d/BiasAdd", "separable_conv2d_1/separable_conv2d"},
                     batch_size);
}

TEST(TestKerasNodes, MaxPool) {
  const std::string pb_path = std::string(tf_root_dir) + "max_pooling.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "max_pooling.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 13, 33, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input"},
                     {"max_pooling2d/MaxPool", "max_pooling2d_1/MaxPool"}, batch_size);
}

TEST(TestKerasNodes, AvgPool) {
  const std::string pb_path = std::string(tf_root_dir) + "average_pooling.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "average_pooling.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 29, 17, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input"},
                     {"average_pooling2d/AvgPool", "average_pooling2d_1/AvgPool"}, batch_size);
}

TEST(TestKerasNodes, FullyConnected) {
  const std::string pb_path = std::string(tf_root_dir) + "dense.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "dense.h5";

  auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {1, 784});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_4"}, {"dense/BiasAdd"});
}

TEST(TestKerasNodes, Permute) {
  const std::string pb_path = std::string(tf_root_dir) + "permute.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "permute.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_8"}, {"permute/transpose"},
                     batch_size);
}

TEST(TestKerasNodes, Reduce) {
  const std::string pb_path = std::string(tf_root_dir) + "reduce.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "reduce.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 24, 24, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input1"},
                     {"tf_op_layer_Mean/Mean", "tf_op_layer_Sum/Sum", "tf_op_layer_Max/Max",
                      "tf_op_layer_Min/Min"},
                     batch_size, 1e-3);
}

TEST(TestKerasNodes, ZeroPadding) {
  const std::string pb_path = std::string(tf_root_dir) + "zero_padding_2d.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "zero_padding_2d.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 12, 24, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"zero_padding2d/Pad", "zero_padding2d_1/Pad", "zero_padding2d_2/Pad",
                      "zero_padding2d_3/Pad"},
                     batch_size);
}

TEST(TestKerasNodes, Conv2DActivation) {
  const std::string pb_path = std::string(tf_root_dir) + "conv2d_activation.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "conv2d_activation.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 23, 29, 3});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"}, {"conv2d/Relu"},
                     batch_size);
}

TEST(TestKerasNodes, Embedding) {
  const std::string pb_path = std::string(tf_root_dir) + "embedding.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "embedding.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomIntTensor<int>(TF_INT32, {batch_size, 10}, 1000);

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input1_1"},
                     {"embedding_4/embedding_lookup/Identity_1"}, batch_size);
}

#ifdef SUPPORT_RNN
TEST(TestKerasNodes, RNNTanh) {
  const std::string pb_path = std::string(tf_root_dir) + "rnn_tanh.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "rnn_tanh.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"simple_rnn/strided_slice_3", "simple_rnn_1/transpose_1"}, batch_size);
}

TEST(TestKerasNodes, RNNRelu) {
  const std::string pb_path = std::string(tf_root_dir) + "rnn_relu.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "rnn_relu.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"simple_rnn/strided_slice_3", "simple_rnn_1/transpose_1"}, batch_size);
}

TEST(TestKerasNodes, BiRNN) {
  const std::string pb_path = std::string(tf_root_dir) + "bidirectional_rnn.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "bidirectional_rnn.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"bidirectional/concat", "bidirectional_1/concat"}, batch_size);
}

TEST(TestKerasNodes, LSTM) {
  const std::string pb_path = std::string(tf_root_dir) + "lstm.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "lstm.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"lstm/strided_slice_7", "lstm_1/transpose_1"}, batch_size, 1e-3);
}

TEST(TestKerasNodes, BiLSTM) {
  const std::string pb_path = std::string(tf_root_dir) + "bidirectional_lstm.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "bidirectional_lstm.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"bidirectional/concat", "bidirectional_1/concat"}, batch_size);
}

TEST(TestKerasNodes, GRU) {
  const std::string pb_path = std::string(tf_root_dir) + "gru.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "gru.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"gru/strided_slice_15", "gru_1/transpose_1"}, batch_size, 0.1);
}

TEST(TestKerasNodes, BiGRU) {
  const std::string pb_path = std::string(tf_root_dir) + "bidirectional_gru.pb";
  const std::string keras_h5_path = std::string(keras_root_dir) + "bidirectional_gru.h5";

  const int batch_size = 1;
  const auto input = fwd::tf_::Utils::CreateRandomTensor<float>(TF_FLOAT, {batch_size, 128, 10});

  TestKerasInference(pb_path, keras_h5_path, {input.get()}, {"input_1"},
                     {"bidirectional/concat", "bidirectional_1/concat"}, batch_size, 0.1);
}
#endif
