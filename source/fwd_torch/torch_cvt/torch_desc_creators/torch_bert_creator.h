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

#include <ATen/core/interned_strings.h>

#include <memory>
#include <string>
#include <vector>

#include "common/fwd_common.h"
#include "common/fwd_utils.h"
#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_module_parser.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief BERT 层描述创建器
 */
template <>
class TLayerDescCreator<TrtBertDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    LOG(INFO) << "TorchBertDesc::Check";

    const auto f_kind = c10::Symbol::fromQualString("fwd::transformer_encoder");

    return node->kind() == f_kind || CheckBertPattern(node);
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TorchBertDesc::Create";
    auto layer_desc = std::make_shared<TrtBertDesc>();

    // Work Mode
    layer_desc->use_fp16 =
        (module.GetMode() == InferMode::HALF) || (module.GetMode() == InferMode::INT8);
    layer_desc->use_int8 =
        (module.GetMode() == InferMode::INT8_CALIB) || (module.GetMode() == InferMode::INT8);
    layer_desc->calib_mode = module.GetMode() == InferMode::INT8_CALIB;

    const auto f_kind = c10::Symbol::fromQualString("fwd::transformer_encoder");

    if (node->kind() == f_kind) {
      if (!CreateBertByFuseOp(node, module, input_values, layer_desc)) return nullptr;
    } else {
      if (!CreateBertByPattern(node, module, input_values, layer_desc)) return nullptr;
    }

    layer_desc->n_layers = SetQKVMatrix(layer_desc.get());

    for (auto& input : input_values) T_CHECK(input);

    return layer_desc;
  }

 private:
  bool CheckBertPattern(const JitNode* node) const {
    // 当前(v1.3.0)Jit模型会把transformer操作拆分成十余节点，需要链式判断节点类型是否吻合
    // 检查 c10::aten::layer_norm 节点
    if (!(node->kind() == c10::aten::layer_norm)) {
      return false;
    }

    auto child = node;
    while (GetLayerIndex(child->inputs()[2]->node()) >= 0) {
      child = child->inputs()[0]->node();  // layer_norm -> add(intermediate)
      if (child->kind() != c10::aten::add && child->kind() != c10::aten::add_) return false;
      child = child->inputs()[1]->node();  // add(intermediate) -> layer_norm
      if (child->kind() != c10::aten::layer_norm) return false;
      child = child->inputs()[0]->node();  // layer_norm -> add(attention)
      if (child->kind() != c10::aten::add && child->kind() != c10::aten::add_) return false;
      child = child->inputs()[1]->node();  // add(attention) -> layer_norm
      if (child->kind() != c10::aten::layer_norm) break;
    }

    if (!(child->kind() == c10::aten::dropout)) return false;

    child = child->inputs()[0]->node();  // dropout -> layer_norm

    return GetLayerIndex(child->inputs()[2]->node()) == -1;
  }

  bool CreateBertByPattern(const JitNode* node, const TorchModule& module,
                           std::vector<const JitValue*>& input_values,
                           std::shared_ptr<TrtBertDesc> layer_desc) {
    LOG(INFO) << "CreateBertByPattern";
    std::unordered_set<const JitNode*> checked_node;
    input_values.resize(3);

    T_CHECK(UpdateLayerInfo(node, module, input_values, layer_desc, checked_node));
    return true;
  }

  bool CreateBertByFuseOp(const JitNode* node, const TorchModule& module,
                          std::vector<const JitValue*>& input_values,
                          std::shared_ptr<TrtBertDesc> layer_desc) {
    LOG(INFO) << "CreateBertByFuseOp: fwd::transformer_encoder";
    int num_layer = 0;
    const auto f_kind = c10::Symbol::fromQualString("fwd::transformer_encoder");
    auto child = node;

    layer_desc->num_heads = module.Get(child->inputs()[18]).toInt();
    layer_desc->hidden_size = module.Get(child->inputs()[20]).toInt();

    while (child->kind() == f_kind) {
      auto inputs = child->inputs();

      // set weights
      for (int i = 1; i < 17; ++i) {
        AnalyzeAttrNode(inputs[i]->node(), module, layer_desc);
      }

      child = inputs[0]->node();  // next layer
      ++num_layer;
    }
    layer_desc->n_layers = num_layer;

    return CreateBertByPattern(child, module, input_values, layer_desc);
  }

  int GetLayerIndex(const JitNode* node) const {
    if (node->inputs().size() < 1) return -2;
    node = node->inputs()[0]->node();  // weight -> LayerNorm
    if (node->kind() != c10::aten::layer_norm && !node->kind().is_prim()) return -2;
    node = node->inputs()[0]->node();  // LayerNorm -> embedding / output
    if (!node->hasAttribute(c10::Symbol::attr("name"))) return -2;
    const std::string attr_name = node->s(c10::Symbol::attr("name"));
    if (attr_name == "embeddings") return -1;

    if (attr_name == "output") {
      node = node->inputs()[0]->node();
      if (!node->hasAttribute(c10::Symbol::attr("name"))) return -2;
      const std::string layer_num = node->s(c10::Symbol::attr("name"));
      return std::stoi(layer_num);
    }
    return -2;
  }

  std::string GetWeightFullName(const JitNode* node) const {
    std::vector<std::string> names;

    auto child = node->inputs()[0]->node();  // weight -> LayerNorm
    const std::string key = node->s(c10::Symbol::attr("name"));
    const bool is_layer_norm = child->s(c10::Symbol::attr("name")) == "LayerNorm";
    if (key == "weight") {
      names.push_back(is_layer_norm ? "gamma" : "kernel");
    } else if (key == "bias") {
      names.push_back(is_layer_norm ? "beta" : "bias");
    }

    while (child->kind() == c10::prim::GetAttr) {
      names.insert(names.begin(), child->s(c10::Symbol::attr("name")));
      child = child->inputs()[0]->node();
    }

    // [0]: namespace(bert); [1]: embedding; [2]: word_embeddings/LayerNorm
    if (names[0].find("bert") != std::string::npos) {
      names.assign(names.begin() + 1, names.end());
    }

    if (names[1].find("embedding") != std::string::npos) {
      names.pop_back();
    }

    std::ostringstream os;
    std::copy(names.begin(), names.end(), std::ostream_iterator<std::string>(os, "_"));
    std::string name = os.str();
    for (auto& c : name) c = std::tolower(c);
    name = FwdUtils::ReplaceAll(name, "embeddings", "embedding");
    return name.substr(0, name.size() - 1);
  }

  bool AnalyzeInput(const JitNode* node, std::vector<const JitValue*>& input_values) const {
    auto node_outputs = node->outputs();
    for (auto input_node : node_outputs) {
      auto node_name = input_node->debugName();

      for (int i = 0; i < 3; i++) {
        if (node_name.find(INPUT_NAMES[i]) != std::string::npos) {
          input_values[i] = input_node;
        }
      }
    }
    return true;
  }

  bool AnalyzeAttrNode(const JitNode* node, const TorchModule& module,
                       std::shared_ptr<TrtBertDesc>& layer_desc) const {
    const std::string key_type = node->s(c10::Symbol::attr("name"));
    if (key_type != "weight" && key_type != "bias") return false;

    const auto weight = module.Get(node->output()).toTensor();
    const std::string layer_name = GetWeightFullName(node);

    // 与tf的kernel存储方式相反，这里应该不需要转置
    layer_desc->weight_map[layer_name] = ToFwdWeights(weight);
    return true;
  }

  void TrySetDesc(const JitNode* node, const TorchModule& module,
                  std::shared_ptr<TrtBertDesc> layer_desc) {
    if (layer_desc->num_heads == 0 && node->kind() == c10::aten::softmax) {
      auto t = module.Get(node->output()).toTensor();
      auto d = DimsOf(t);
      layer_desc->num_heads = d.d[1];
    }

    if (node->kind() == c10::aten::relu) {
      layer_desc->use_relu = true;
    }
  }

  bool UpdateLayerInfo(const JitNode* node, const TorchModule& module,
                       std::vector<const JitValue*>& input_values,
                       std::shared_ptr<TrtBertDesc>& layer_desc,
                       std::unordered_set<const JitNode*>& checked_node) {
    if (checked_node.find(node) != checked_node.end()) {
      return true;
    }
    checked_node.insert(node);

    TrySetDesc(node, module, layer_desc);

    auto layer_type = node->kind();

    if (layer_type == c10::prim::GetAttr) {
      AnalyzeAttrNode(node, module, layer_desc);
      UpdateLayerInfo(node->inputs()[0]->node(), module, input_values, layer_desc, checked_node);
      return true;
    }

    if (node->inputs().empty()) {
      AnalyzeInput(node, input_values);
      return true;
    }

    for (auto input : node->inputs()) {
      UpdateLayerInfo(input->node(), module, input_values, layer_desc, checked_node);
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

  const std::vector<const char*> INPUT_NAMES{"input_ids", "input", "attention_mask"};

  const std::unordered_map<std::string, std::string> NAME_MAP{
      // Attention Keys
      {"AT_WQ", "query_kernel"}, {"AT_BQ", "query_bias"},   {"AT_WK", "key_kernel"},
      {"AT_BK", "key_bias"},     {"AT_WV", "value_kernel"}, {"AT_BV", "value_bias"},
  };
};

FWD_TORCH_NAMESPACE_END
