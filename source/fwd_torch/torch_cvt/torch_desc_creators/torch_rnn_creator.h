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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief RNN 层描述创建器
 */
template <>
class TLayerDescCreator<TrtRNNv2Desc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    const auto kind = node->kind();

    return kind == c10::aten::rnn_relu || kind == c10::aten::rnn_tanh || kind == c10::aten::lstm ||
           kind == c10::aten::gru;
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtRNNv2Desc::Create";

    auto layer_desc = std::make_shared<TrtRNNv2Desc>();

    layer_desc->torchOrder = true;

    const auto kind = node->kind();
    layer_desc->operation = NK2RNNOP_MAPPING.find(kind)->second;

    const auto inputs = node->inputs();
    input_values.push_back(inputs[0]);

    // input 0 Tensor& input,
    // input 1 const Tensor& hx, (TensorList for LSTM)
    // input 2 TensorList params,
    // input 3 bool has_biases,
    // input 4 int64_t num_layers,
    // input 5 double dropout,
    // input 6 bool train,
    // input 7 bool bidirectional,
    // input 8 bool batch_first

    const auto params = module.Get(inputs[2]).toTensorList();
    const auto has_biases = module.Get(inputs[3]).toBool();
    const auto num_layers = module.Get(inputs[4]).toInt();
    const auto bidirectional = module.Get(inputs[7]).toBool();
    const auto batch_first = module.Get(inputs[8]).toBool();
    layer_desc->batchFirst = batch_first;

    const auto trt_type = ConvertTorch2TrtDtype(params.get(0).scalar_type());

    // params 排布顺序 : layer0
    //     : input_weights,
    //         hidden_weights,
    //         (input_bias, hidden_bias, )(
    //             input_weights_reverse, hidden_weights_reverse,
    //             (input_bias_reverse, hidden_bias_reverse));
    // layer1:
    //   ... 对应门的信息目前在create network时解包

    int64_t i = 0;
    while (i < params.size()) {
      for (int j = 0; j < 1 + bidirectional; j++) {
        layer_desc->weightsForGate.push_back(ToFwdWeights(params.get(i)));
        layer_desc->weightsForGate.push_back(ToFwdWeights(params.get(i + 1)));
        i += 2;
        if (has_biases) {
          layer_desc->biasForGate.push_back(ToFwdWeights(params.get(i)));
          layer_desc->biasForGate.push_back(ToFwdWeights(params.get(i + 1)));
          i += 2;
        }
      }
    }

    layer_desc->direction = bidirectional ? nvinfer1::RNNDirection::kBIDIRECTION
                                          : nvinfer1::RNNDirection::kUNIDIRECTION;
    layer_desc->layerCount = num_layers;

    // 这里获取到的State是无意义的:
    // hiddenSize只能从这里拿到
    // TODO(Paul Lu): 通过CheckZeros判断是否从Input获取hidden
    // State，有潜在的判断错误风险
    if (layer_desc->operation == nvinfer1::RNNOperation::kLSTM) {
      const auto hx = module.Get(inputs[1]).toTensorList();
      layer_desc->hiddenSize = hx.get(0).sizes()[2];
      if (!(CheckZeros(hx.get(0)) && CheckZeros(hx.get(1)))) {
        input_values.push_back(inputs[1]);
      }
    } else {
      const auto hx = module.Get(inputs[1]).toTensor();
      layer_desc->hiddenSize = hx.sizes()[2];
      if (!CheckZeros(hx)) {
        input_values.push_back(inputs[1]);
      }
    }

    return layer_desc;
  }

 private:
  const std::unordered_map<c10::Symbol, nvinfer1::RNNOperation> NK2RNNOP_MAPPING = {
      {c10::aten::rnn_relu, nvinfer1::RNNOperation::kRELU},
      {c10::aten::rnn_tanh, nvinfer1::RNNOperation::kTANH},
      {c10::aten::lstm, nvinfer1::RNNOperation::kLSTM},
      {c10::aten::gru, nvinfer1::RNNOperation::kGRU}};

  static bool CheckZeros(const at::Tensor& tensor) {
    auto* data = static_cast<float*>(tensor.data_ptr());
    for (size_t i = 0; i < tensor.numel(); i++) {
      if (data[i] != 0) {
        return false;
      }
    }
    return true;
  }
};

FWD_TORCH_NAMESPACE_END
