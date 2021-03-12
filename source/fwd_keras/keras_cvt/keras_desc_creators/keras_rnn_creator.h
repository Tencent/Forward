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

#include "fwd_keras/keras_cvt/keras_desc_creators/i_keras_layer_creator.h"

FWD_KERAS_NAMESPACE_BEGIN
/**
 * \brief RNN DescCreator
 */
template <>
class TLayerDescCreator<TrtRNNv2Desc> : public ILayerDescCreator {
 public:
  bool Check(const Layer& layer) override {
    // TODO(Ao Li): 这里假设 Bidirectional 只会出现在 RNN 系列层中
    const std::string type = layer.Type();
    return type == "Bidirectional" || NAME2OP_MAPPING.find(type) != NAME2OP_MAPPING.end();
  }

  std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                       std::vector<std::string>& input_names) override {
    LOG(INFO) << "TrtRNNv2Desc::Create";

    input_names = layer.Inputs();

    const std::string layer_name = layer.Name();

    std::string layer_type_name = layer.Type();

    Layer new_layer = layer;
    if (layer_type_name == "Bidirectional") {
      const std::string layer_name = new_layer.Name();
      group_prefix_ = layer_name + "/" + layer_name + "/";
      bidirectional_ = true;
      const std::string merge_mode = new_layer.GetAttr<std::string>("merge_mode");
      if (merge_mode != "concat") {
        LOG(ERROR) << "Unsupported merge mode " << merge_mode << " in bidirectional "
                   << layer_type_name;
        return nullptr;
      }
      new_layer = Layer(new_layer.GetAttr<json>("layer"));
      layer_type_name = new_layer.Type();
    }

    auto layer_type = NAME2OP_MAPPING.find(layer_type_name);

    if (layer_type != NAME2OP_MAPPING.end()) {
      return CreateRNN(new_layer, reader, layer_type->second);
    }

    return nullptr;
  }

 private:
  std::shared_ptr<TrtLayerDesc> CreateRNN(const Layer& layer, const H5ModelReader& reader,
                                          nvinfer1::RNNOperation op_type) {
    const std::string config_name = layer.GetAttr<std::string>("name");
    std::shared_ptr<TrtLayerDesc> default_return_value = nullptr;

    // TODO(Ao Li): 支持更多的属性
    const std::string activation = layer.GetAttr<std::string>("activation");
    const bool return_sequences = layer.GetAttr<bool>("return_sequences");
    const bool use_bias = layer.GetAttr<bool>("use_bias");

    if (activation != "tanh") {
      if (op_type == nvinfer1::RNNOperation::kTANH && activation == "relu") {
        op_type = nvinfer1::RNNOperation::kRELU;
      } else {
        LOG(ERROR) << "Unsupported activation \"" << activation << "\"";
        return nullptr;
      }
    }

    if (layer.HasAttr("recurrent_activation")) {
      T_CHECK_EQ(layer.GetAttr<std::string>("recurrent_activation"), "sigmoid");
    }

    auto layer_desc = std::make_shared<TrtRNNv2Desc>();

    layer_desc->hiddenSize = layer.GetAttr<int>("units");
    layer_desc->layerCount = 1;

    layer_desc->operation = op_type;
    layer_desc->inputMode = nvinfer1::RNNInputMode::kLINEAR;
    layer_desc->returnSequences = return_sequences;

    if (bidirectional_) {
      layer_desc->direction = nvinfer1::RNNDirection::kBIDIRECTION;
      auto forward_input_weights =
          reader.ReadWeight(group_prefix_ + "forward_" + config_name, "kernel:0");
      auto forward_hidden_weights =
          reader.ReadWeight(group_prefix_ + "forward_" + config_name, "recurrent_kernel:0");
      auto backward_input_weights =
          reader.ReadWeight(group_prefix_ + "backward_" + config_name, "kernel:0");
      auto backward_hidden_weights =
          reader.ReadWeight(group_prefix_ + "backward_" + config_name, "recurrent_kernel:0");
      layer_desc->weightsForGate.push_back(forward_input_weights.ToFwdWeights());
      layer_desc->weightsForGate[0].Transpose(forward_input_weights.GetDimension(), {1, 0});
      layer_desc->weightsForGate.push_back(forward_hidden_weights.ToFwdWeights());
      layer_desc->weightsForGate[1].Transpose(forward_hidden_weights.GetDimension(), {1, 0});
      layer_desc->weightsForGate.push_back(backward_input_weights.ToFwdWeights());
      layer_desc->weightsForGate[2].Transpose(backward_input_weights.GetDimension(), {1, 0});
      layer_desc->weightsForGate.push_back(backward_hidden_weights.ToFwdWeights());
      layer_desc->weightsForGate[3].Transpose(backward_hidden_weights.GetDimension(), {1, 0});
      if (use_bias) {
        auto forward_input_bias =
            reader.ReadWeight(group_prefix_ + "forward_" + config_name, "bias:0");
        auto backward_input_bias =
            reader.ReadWeight(group_prefix_ + "backward_" + config_name, "bias:0");
        // 可分离 bias
        if (forward_input_bias.GetDimension().nbDims == 2) {
          T_CHECK_EQ(forward_input_bias.GetDimension().d[0], 2);
          T_CHECK_EQ(backward_input_bias.GetDimension().d[0], 2);
          layer_desc->biasForGate.push_back(forward_input_bias.GetRow(0).ToFwdWeights());
          layer_desc->biasForGate.push_back(forward_input_bias.GetRow(1).ToFwdWeights());
          layer_desc->biasForGate.push_back(backward_input_bias.GetRow(0).ToFwdWeights());
          layer_desc->biasForGate.push_back(backward_input_bias.GetRow(1).ToFwdWeights());
        } else {
          T_CHECK_EQ(forward_input_bias.GetDimension().nbDims, 1);
          T_CHECK_EQ(backward_input_bias.GetDimension().nbDims, 1);
          layer_desc->biasForGate.push_back(
              FwdWeights(std::vector<float>(forward_input_bias.GetCount(), 0)));
          layer_desc->biasForGate.push_back(forward_input_bias.ToFwdWeights());
          layer_desc->biasForGate.push_back(
              FwdWeights(std::vector<float>(backward_input_bias.GetCount(), 0)));
          layer_desc->biasForGate.push_back(backward_input_bias.ToFwdWeights());
        }
      }
    } else {
      layer_desc->direction = nvinfer1::RNNDirection::kUNIDIRECTION;
      auto input_weights = reader.ReadWeight(config_name, "kernel:0");
      auto hidden_weights = reader.ReadWeight(config_name, "recurrent_kernel:0");
      layer_desc->weightsForGate.push_back(input_weights.ToFwdWeights());
      layer_desc->weightsForGate[0].Transpose(input_weights.GetDimension(), {1, 0});
      layer_desc->weightsForGate.push_back(hidden_weights.ToFwdWeights());
      layer_desc->weightsForGate[1].Transpose(hidden_weights.GetDimension(), {1, 0});
      if (use_bias) {
        auto input_bias = reader.ReadWeight(config_name, "bias:0");
        // 可分离 bias
        if (input_bias.GetDimension().nbDims == 2) {
          T_CHECK_EQ(input_bias.GetDimension().d[0], 2);
          layer_desc->biasForGate.push_back(input_bias.GetRow(0).ToFwdWeights());
          layer_desc->biasForGate.push_back(input_bias.GetRow(1).ToFwdWeights());
        } else {
          T_CHECK_EQ(input_bias.GetDimension().nbDims, 1);
          layer_desc->biasForGate.push_back(
              FwdWeights(std::vector<float>(input_bias.GetCount(), 0)));
          layer_desc->biasForGate.push_back(input_bias.ToFwdWeights());
        }
      }
    }

    return layer_desc;
  }

  bool bidirectional_ = false;

  std::string group_prefix_ = "";

  const std::unordered_map<std::string, nvinfer1::RNNOperation> NAME2OP_MAPPING = {
      {"SimpleRNN", nvinfer1::RNNOperation::kTANH},  // 0
      {"LSTM", nvinfer1::RNNOperation::kLSTM},       // 1
      {"GRU", nvinfer1::RNNOperation::kGRU},         // 2
  };
};

FWD_KERAS_NAMESPACE_END
