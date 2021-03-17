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
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN
/**
 * \brief Bert 层描述创建器
 */
template <>
class TLayerDescCreator<TrtBertDesc> : public ILayerDescCreator {
 public:
  bool Check(const Operation& op) override {
    const std::string name = op.Name();

    std::smatch match;
    return std::regex_search(name, match, std::regex(BERT_PREFIX + "encoder"));
  }

  std::shared_ptr<TrtLayerDesc> Create(const Operation& op, const Graph& graph,
                                       std::vector<Output>& op_inputs) override {
    LOG(INFO) << "TrtBertDesc::Create";
    Operation father = op;  // Sum Node or Gather Node

    // ignore the last reshape layer
    while (father.Name().find("encoder/layer") == std::string::npos) {
      father = father.Input(0);
    }

    const std::string name = father.Name();
    std::regex e("encoder/layer_([0-9]+)");
    std::smatch m;

    T_CHECK(std::regex_search(name, m, e));

    const int n_layer = std::stoi(m.str(1)) + 1;

    auto layer_desc = std::make_shared<TrtBertDesc>();
    layer_desc->n_layers = n_layer;
    layer_desc->use_fp16 = graph.Mode() == InferMode::HALF || graph.Mode() == InferMode::INT8;
    layer_desc->use_int8 = graph.Mode() == InferMode::INT8_CALIB || graph.Mode() == InferMode::INT8;
    layer_desc->calib_mode = graph.Mode() == InferMode::INT8_CALIB;

    std::unordered_set<const TF_Operation*> checked_op;
    op_inputs.resize(3);

    T_CHECK(UpdateLayerInfo(father, graph, op_inputs, layer_desc, checked_op));

    const int n_layer_check = SetQKVMatrix(layer_desc.get());
    T_CHECK_EQ(n_layer_check, n_layer);

    // 检查是否有输入未找到
    for (auto& input : op_inputs) T_CHECK(input.Op());

    return layer_desc;
  }

 private:
  void TrySetDesc(const Operation& op, std::shared_ptr<TrtBertDesc>& layer_desc,
                  const std::string& name) const {
    // Get num_heads from SOFTMAX node
    if (name.find(NAME_MAP.at("SOFTMAX")) != std::string::npos) {
      const auto op_shape = DimsOf(Output{op.Graph(), op.Op(), 0});
      layer_desc->num_heads = op_shape.d[1];
    }

    // 判断bert是否用了relu
    if (name.find("Relu") != std::string::npos) {
      layer_desc->use_relu = true;
    }

    // 判断bert是否用了 group conv1d
    if (name.find("grouped_convolution_") != std::string::npos) {
      layer_desc->use_group_conv1d = true;
    }

    // 获取 intermediate Reshape 的 shape 参数
    if (op.OpType() == "Reshape" && name.find("intermediate/Reshape") != std::string::npos) {
      layer_desc->intermediate_shape =
          TrtUtils::ToDims(op.Input(1).GetConstantTensor().AsIntList());
    }

    // 获取 num_split 参数，这里认为 intermediate 和 output 有相同的 num_split
    if (op.OpType() == "Split" && (name.find("output/split") != std::string::npos ||
                                   name.find("intermediate/split") != std::string::npos)) {
      layer_desc->num_split = op.GetAttrInt("num_split");
    }
  }

  void SetWeightData(const Operation& op, std::shared_ptr<TrtBertDesc>& layer_desc,
                     const std::string& name) const {
    // 把名字中的斜杠替换为下划线，字母统一转为小写
    auto layer_name = FwdUtils::ReplaceAll(name, "/", "_");
    layer_name = FwdUtils::ReplaceAll(layer_name, "embeddings", "embedding");
    for (auto& c : layer_name) c = std::tolower(c);

    const auto output = Output(op.Graph(), op.Op(), 0);
    const auto tf_weight = output.GetConstantTensor();
    layer_desc->weight_map[layer_name] = ToFwdWeights(tf_weight);

    const int offset = layer_name.find("layernorm");
    if (offset > 0) {
      const std::string bias_name = layer_name.substr(0, offset) + "layernorm_bias";
      std::vector<float> data(tf_weight.ElementCount());
      layer_desc->weight_map[bias_name] = FwdWeights(data);
    }

    // 对Kernel类的数据做转置，把未转置的数据放在后面
    if (layer_name.find("kernel") != std::string::npos) {
      layer_desc->weight_map[layer_name + "_notrans"] = ToFwdWeights(tf_weight);

      layer_desc->weight_map[layer_name].Transpose(DimsOf(output), {1, 0});
    }

    // 对Conv1d卷积参数进行转置，[H,W,in,out]->[out,in,H,W]
    if (layer_name.find("conv1d_expanddims_1") != std::string::npos) {
      FwdWeights& weights = layer_desc->weight_map[layer_name];
      weights.Transpose(weights.Dims(), {3, 2, 0, 1});
      nvinfer1::Dims dims = weights.Dims();
      dims.d[0] = dims.d[3];
      dims.d[1] = dims.d[2];
      dims.d[2] = dims.d[0];
      dims.d[3] = dims.d[1];
      weights.SetDims(dims);
    }
  }

  bool SetInput(const Operation& op, std::vector<Output>& op_inputs,
                std::unordered_set<const TF_Operation*>& checked_op, const std::string name) const {
    for (auto i = 0u; i < INPUT_NAMES.size(); ++i) {
      if (name.find(INPUT_NAMES[i]) != std::string::npos) {
        op_inputs[i] = Output(op.Graph(), op.Op(), 0);
        checked_op.insert(op.Op());
        return true;
      }
    }
    return false;
  }

  bool UpdateLayerInfo(const Operation& op, const Graph& graph, std::vector<Output>& op_inputs,
                       std::shared_ptr<TrtBertDesc>& layer_desc,
                       std::unordered_set<const TF_Operation*>& checked_op) const {
    if (checked_op.find(op.Op()) != checked_op.end()) return true;

    const std::string name = op.Name();

    TrySetDesc(op, layer_desc, name);

    // if bert_prefix not found, mark input
    std::smatch match;

    checked_op.insert(op.Op());

    if (!std::regex_search(name, match, std::regex(BERT_PREFIX))) {
      return SetInput(op, op_inputs, checked_op, name);
    }

    for (auto& pair : NAME_MAP) {
      if (std::regex_search(name, match, std::regex(pair.second))) {
        SetWeightData(op, layer_desc, match.str(0));
      }
    }

    for (int i = 0; i < op.NumInputs(); i++) {
      if (!UpdateLayerInfo(op.Input(i), graph, op_inputs, layer_desc, checked_op)) return false;
    }

    return true;
  }

  int SetQKVMatrix(TrtBertDesc* desc) const {
    int count = 0;
    auto& weight_map = desc->weight_map;

    std::vector<std::string> keys;
    keys.reserve(weight_map.size());
    for (auto& entry : weight_map) keys.push_back(entry.first);

    for (auto& key : keys) {
      const size_t pos = key.find(NAME_MAP.at("AT_BQ"));  // starting pos of BQ

      if (pos == std::string::npos) continue;

      count++;
      const std::string prefix = key.substr(0, pos);
      const auto& q_b = weight_map.at(key);

      const int hidden_size = q_b.Count();
      desc->hidden_size = hidden_size;

      const auto& k_b = weight_map.at(prefix + NAME_MAP.at("AT_BK"));
      const auto& v_b = weight_map.at(prefix + NAME_MAP.at("AT_BV"));
      const auto& q_w = weight_map.at(prefix + NAME_MAP.at("AT_WQ"));
      const auto& k_w = weight_map.at(prefix + NAME_MAP.at("AT_WK"));
      const auto& v_w = weight_map.at(prefix + NAME_MAP.at("AT_WV"));

      const int mat_size = hidden_size * hidden_size;
      std::vector<float> qkv_w(3 * mat_size);
      std::vector<float> qkv_b(3 * hidden_size);

      q_w.CopyTo(qkv_w.data(), mat_size);
      k_w.CopyTo(qkv_w.data() + mat_size, mat_size);
      v_w.CopyTo(qkv_w.data() + 2 * mat_size, mat_size);
      q_b.CopyTo(qkv_b.data(), hidden_size);
      k_b.CopyTo(qkv_b.data() + hidden_size, hidden_size);
      v_b.CopyTo(qkv_b.data() + 2 * hidden_size, hidden_size);

      auto qkv_weights = FwdWeights(qkv_w);
      auto qkv_biases = FwdWeights(qkv_b);
      qkv_weights.Transpose({5, 3, desc->num_heads, desc->hidden_size / desc->num_heads,
                             desc->num_heads, desc->hidden_size / desc->num_heads},
                            {1, 0, 2, 3, 4});
      qkv_biases.Transpose({3, 3, desc->num_heads, desc->hidden_size / desc->num_heads}, {1, 0, 2});

      weight_map[prefix + TrtBertDesc::WQKV] = qkv_weights;
      weight_map[prefix + TrtBertDesc::BQKV] = qkv_biases;
    }

    return count;
  }

  std::string BERT_PREFIX = "^((tiny/)|(bert/)|(bert/bert/))";

  const std::vector<const char*> INPUT_NAMES{"input_ids", "segment_ids", "input_mask"};

  const std::unordered_map<std::string, std::string> NAME_MAP{
      {"WQ", "encoder/layer_([0-9]+)/attention/self/query/kernel$"},
      {"BQ", "encoder/layer_([0-9]+)/attention/self/query/bias$"},
      {"WK", "encoder/layer_([0-9]+)/attention/self/key/kernel$"},
      {"BK", "encoder/layer_([0-9]+)/attention/self/key/bias$"},
      {"WV", "encoder/layer_([0-9]+)/attention/self/value/kernel$"},
      {"BV", "encoder/layer_([0-9]+)/attention/self/value/bias$"},
      {"WQKV", "reserved"},
      {"BQKV", "reserved"},
      {"W_AOUT", "encoder/layer_([0-9]+)/attention/output/dense/kernel$"},
      {"B_AOUT", "encoder/layer_([0-9]+)/attention/output/dense/bias$"},
      {"AOUT_LN_BETA", "encoder/layer_([0-9]+)/attention/output/LayerNorm/beta$"},
      {"AOUT_LN_GAMMA", "encoder/layer_([0-9]+)/attention/output/LayerNorm/gamma$"},
      {"W_MID", "encoder/layer_([0-9]+)/intermediate/dense/kernel$"},
      {"B_MID", "encoder/layer_([0-9]+)/intermediate/dense/bias$"},
      {"W_LOUT", "encoder/layer_([0-9]+)/output/dense/kernel$"},
      {"B_LOUT", "encoder/layer_([0-9]+)/output/dense/bias$"},
      {"LOUT_LN_BETA", "encoder/layer_([0-9]+)/output/LayerNorm/beta$"},
      {"LOUT_LN_GAMMA", "encoder/layer_([0-9]+)/output/LayerNorm/gamma$"},
      {"EMB_LN_BETA", "embedding([s])?/LayerNorm/beta$"},
      {"EMB_LN_GAMMA", "embedding([s])?/LayerNorm/gamma$"},
      {"EMB_WORD_EMBEDDINGS", "embedding([s])?/word_embeddings$"},
      {"EMB_TT_EMBEDDINGS", "embedding([s])?/token_type_embeddings$"},
      {"EMB_POS_EMBEDDINGS", "embedding([s])?/position_embeddings$"},

      {"SOFTMAX", "encoder/layer_0/attention/self/Softmax"},

      // Attention Keys
      {"AT_WQ", "query_kernel"},
      {"AT_BQ", "query_bias"},
      {"AT_WK", "key_kernel"},
      {"AT_BK", "key_bias"},
      {"AT_WV", "value_kernel"},
      {"AT_BV", "value_bias"},

      // For Grouped Conv1D
      {"CONV_IW",
       "encoder/layer_([0-9]+)/intermediate/grouped_convolution_([0-9]+)/"
       "conv1d/ExpandDims_1$"},
      {"CONV_OW",
       "encoder/layer_([0-9]+)/output/grouped_convolution_([0-9]+)/conv1d/"
       "ExpandDims_1$"},
      {"CONV_IB",
       "encoder/layer_([0-9]+)/intermediate/grouped_convolution_([0-9]+)/bias/"
       "read$"},
      {"CONV_OB", "encoder/layer_([0-9]+)/output/grouped_convolution_([0-9]+)/bias/read$"},
  };
};

FWD_TF_NAMESPACE_END
