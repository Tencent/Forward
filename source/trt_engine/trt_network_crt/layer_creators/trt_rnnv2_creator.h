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

#include <vector>

#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT RNNv2层创建器
 * 如果网络中使用了RNNv2层，那么Forward时需调用V1版本的api，如execute/enqueue.
 * 这是一个已知bug，会在未来的TensorRT中修复。
 */
template <>
class TLayerCreator<TrtRNNv2Desc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtRNNv2Desc::CreateLayer";

    const auto rnn_desc = dynamic_cast<const TrtRNNv2Desc*>(layer_desc);
    T_CHECK(rnn_desc);

    auto& input = *input_tensors[0];

    auto input_dim = input.getDimensions();

    // 如果!batchFirst需要对输入转置
    if (!rnn_desc->batchFirst) {
      auto transpose_input = network->addShuffle(input);
      transpose_input->setFirstTranspose({1, 0, 2});
      input = *transpose_input->getOutput(0);
      input_dim = input.getDimensions();
    }

    auto batch_size = input_dim.d[0];
    const auto seq_len = input_dim.d[1];
    const auto input_size = input_dim.d[2];
    const auto hidden_size = rnn_desc->hiddenSize;
    nvinfer1::IRNNv2Layer* rnn =
        network->addRNNv2(input, rnn_desc->layerCount, hidden_size, seq_len, rnn_desc->operation);

    if (rnn == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [rnnv2] layer.";
      return {};
    }

    rnn->setDirection(rnn_desc->direction);

    // set Weights
    const bool is_bidirection = (rnn_desc->direction == nvinfer1::RNNDirection::kBIDIRECTION);
    const auto& gate_types = OP2GATE_MAPPING[static_cast<int>(rnn_desc->operation)];
    for (int i = 0; i < rnn_desc->layerCount; i++) {
      auto index = is_bidirection ? 2 * i : i;
      const auto& iw = rnn_desc->weightsForGate[i * 2 * (1 + is_bidirection)];
      const auto& hw = rnn_desc->weightsForGate[i * 2 * (1 + is_bidirection) + 1];

      // TODO(Ao Li): 是否要考虑不是float的情形？
      auto* iw_ptr = static_cast<const float*>(iw.Data());
      auto* hw_ptr = static_cast<const float*>(hw.Data());

      const auto iw_count = index == 0 ? (input_size * hidden_size)
                                       : (hidden_size * hidden_size * (1 + is_bidirection));
      const auto hw_count = hidden_size * hidden_size;

      for (int j = 0; j < gate_types.size(); j++) {
        const nvinfer1::Weights weights1 = {iw.Type(), iw_ptr + j * iw_count, iw_count};
        rnn->setWeightsForGate(index, gate_types[j], true, weights1);

        const nvinfer1::Weights weights2 = {hw.Type(), hw_ptr + j * hw_count, hw_count};
        rnn->setWeightsForGate(index, gate_types[j], false, weights2);
      }
      if (!rnn_desc->biasForGate.empty()) {
        const auto& ib = rnn_desc->biasForGate[i * 2 * (1 + is_bidirection)];
        const auto& hb = rnn_desc->biasForGate[i * 2 * (1 + is_bidirection) + 1];

        auto* ib_ptr = static_cast<const float*>(ib.Data());
        auto* hb_ptr = static_cast<const float*>(hb.Data());
        for (int j = 0; j < gate_types.size(); j++) {
          const nvinfer1::Weights bias1 = {ib.Type(), ib_ptr + j * hidden_size, hidden_size};
          rnn->setBiasForGate(index, gate_types[j], true, bias1);

          const nvinfer1::Weights bias2 = {hb.Type(), hb_ptr + j * hidden_size, hidden_size};
          rnn->setBiasForGate(index, gate_types[j], false, bias2);
        }
      }
      if (is_bidirection) {
        index = 2 * i + 1;
        const auto& iwr = rnn_desc->weightsForGate[i * 4 + 2];
        const auto& hwr = rnn_desc->weightsForGate[i * 4 + 3];

        auto* iwr_ptr = static_cast<const float*>(iwr.Data());
        auto* hwr_ptr = static_cast<const float*>(hwr.Data());
        for (int j = 0; j < gate_types.size(); j++) {
          const nvinfer1::Weights weights1 = {iwr.Type(), iwr_ptr + j * iw_count, iw_count};
          rnn->setWeightsForGate(index, gate_types[j], true, weights1);

          const nvinfer1::Weights weights2 = {hwr.Type(), hwr_ptr + j * hw_count, hw_count};
          rnn->setWeightsForGate(index, gate_types[j], false, weights2);
        }
        if (!rnn_desc->biasForGate.empty()) {
          const auto& ibr = rnn_desc->biasForGate[i * 4 + 2];
          const auto& hbr = rnn_desc->biasForGate[i * 4 + 3];

          auto* ibr_ptr = static_cast<const float*>(ibr.Data());
          auto* hbr_ptr = static_cast<const float*>(hbr.Data());
          for (int j = 0; j < gate_types.size(); j++) {
            const nvinfer1::Weights bias1 = {ibr.Type(), ibr_ptr + j * hidden_size, hidden_size};
            rnn->setBiasForGate(index, gate_types[j], true, bias1);

            const nvinfer1::Weights bias2 = {hbr.Type(), hbr_ptr + j * hidden_size, hidden_size};
            rnn->setBiasForGate(index, gate_types[j], false, bias2);
          }
        }
      }
    }

    // Set initial hidden state (and cell state for LSTM)
    // torch输入为(layer_count*direction_count, batch_size, hidden_size),
    // 需要转置
    if (input_tensors.size() > 1) {
      if (CheckBatchPos(input_tensors[1], batch_size)) {
        auto hidden_state_layer = network->addShuffle(*input_tensors[1]);
        hidden_state_layer->setFirstTranspose({1, 0, 2});
        rnn->setHiddenState(*hidden_state_layer->getOutput(0));
      } else {
        rnn->setHiddenState(*input_tensors[1]);
      }

      if (rnn_desc->operation == nvinfer1::RNNOperation::kLSTM) {
        if (CheckBatchPos(input_tensors[2], batch_size)) {
          auto cell_state_layer = network->addShuffle(*input_tensors[2]);
          cell_state_layer->setFirstTranspose({1, 0, 2});
          rnn->setCellState(*cell_state_layer->getOutput(0));
        } else {
          rnn->setCellState(*input_tensors[2]);
        }
      }
    }

    LOG(INFO) << "rnn output dims =" << TrtUtils::ShapeStrOf(rnn->getOutput(0)->getDimensions());

    if (rnn_desc->returnSequences) {
      return {rnn->getOutput(0), rnn->getOutput(1), rnn->getOutput(2)};
    }

    // 返回 t 时刻的序列
    const auto dims = rnn->getOutput(0)->getDimensions();
    T_CHECK_EQ(dims.nbDims, 3);

    // 单向直接返回最后一行
    if (rnn_desc->direction == nvinfer1::RNNDirection::kUNIDIRECTION) {
      const nvinfer1::Dims start = {3, {0, dims.d[1] - 1, 0}};
      const nvinfer1::Dims size = {3, {dims.d[0], 1, dims.d[2]}};
      const nvinfer1::Dims stride = {3, {1, 1, 1}};
      const auto slice_layer = network->addSlice(*rnn->getOutput(0), start, size, stride);
      return {slice_layer->getOutput(0)};
    }

    // 双向拼接最后一行的前半部分和第一行的后半部分
    const nvinfer1::Dims start1 = {3, {0, dims.d[1] - 1, 0}};
    const nvinfer1::Dims size1 = {3, {dims.d[0], 1, dims.d[2] / 2}};
    const nvinfer1::Dims stride1 = {3, {1, 1, 1}};
    nvinfer1::ISliceLayer* slice1 = network->addSlice(*rnn->getOutput(0), start1, size1, stride1);

    const nvinfer1::Dims start2 = {3, {0, 0, dims.d[2] / 2}};
    const nvinfer1::Dims size2 = {3, {dims.d[0], 1, dims.d[2] / 2}};
    const nvinfer1::Dims dims2 = {3, {1, 1, 1}};
    nvinfer1::ISliceLayer* slice2 = network->addSlice(*rnn->getOutput(0), start2, size2, dims2);

    ITensorVector inputs{slice1->getOutput(0), slice2->getOutput(0)};
    const auto cat = network->addConcatenation(inputs.data(), inputs.size());
    cat->setAxis(2);
    return {cat->getOutput(0)};
  }

 private:
  static int CheckBatchPos(const nvinfer1::ITensor* tensor, int batch_size) {
    const auto tensor_dims = tensor->getDimensions();
    for (int i = 0; i < tensor_dims.nbDims; i++) {
      if (tensor_dims.d[i] == batch_size) {
        return i;
      }
    }
    return -1;
  }

  static const std::vector<std::vector<nvinfer1::RNNGateType>> OP2GATE_MAPPING;
};

const std::vector<std::vector<nvinfer1::RNNGateType>> TLayerCreator<TrtRNNv2Desc>::OP2GATE_MAPPING =
    {
        {nvinfer1::RNNGateType::kINPUT},  // RNNOperation::kRELU, 0
        {nvinfer1::RNNGateType::kINPUT},  // RNNOperation::kTANH, 1
        {nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kFORGET,
         nvinfer1::RNNGateType::kCELL, nvinfer1::RNNGateType::kOUTPUT},  // RNNOperation::kLSTM, 2
        {nvinfer1::RNNGateType::kRESET, nvinfer1::RNNGateType::kUPDATE,
         nvinfer1::RNNGateType::kHIDDEN},  // RNNOperation::kGRU, 3
};

FWD_TRT_NAMESPACE_END
