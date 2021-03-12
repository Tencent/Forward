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

#include "fwd_keras/keras_cvt/keras_cpp_api.h"

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

FWD_KERAS_NAMESPACE_BEGIN

bool H5ModelReader::LoadModel(const std::string& model_path) {
  file_.openFile(model_path, H5F_ACC_RDONLY);
  model_weights_ = file_.openGroup("model_weights");

  /* TODO(yzx):
   * Keras 现在有两种 模型 保存格式：
   * 1. Model 类型，可以递归解析；
   * 2. Sequential 类型，需要顺序解析
   *
   * 目前先解析 Model 类型
   */
  json model = ReadModelConfig();
  if (model.empty() || model.at("class_name").get<std::string>() != "Model") return false;

  model["config"] = model_preprocessor_.OptimizeConfig(model.at("config"));

  ExtractModelInfos(model.at("config"));

  return true;
}

json H5ModelReader::ReadModelConfig() const {
  static const std::string MODEL_CONFIG_ATTR_NAME = "model_config";
  std::string model_config;

  if (!file_.attrExists(MODEL_CONFIG_ATTR_NAME)) {
    LOG(ERROR) << "Invalid keras model: no model_config attribute found.";
  } else {
    H5::Attribute model_config_attr = file_.openAttribute(MODEL_CONFIG_ATTR_NAME);
    model_config_attr.read(model_config_attr.getDataType(), model_config);
    model_config_attr.close();
  }

  return json::parse(model_config);
}

KerasWeights H5ModelReader::ReadWeight(const std::string& group_name,
                                       const std::string& weight_name) const {
  H5::Group group = model_weights_.openGroup(group_name);
  // 找到 data set
  while (group.getObjTypeByIdx(0) == H5G_GROUP) {
    auto temp = group;
    CHECK_EQ(group.getNumObjs(), 1);
    group = group.openGroup(group.getObjnameByIdx(0));
    temp.close();
  }

  auto dataset = group.openDataSet(weight_name);
  auto dataspace = dataset.getSpace();

  const int rank = dataspace.getSimpleExtentNdims();

  std::vector<hsize_t> dims(rank);

  CHECK_EQ(dataspace.getSimpleExtentDims(dims.data()), rank);
  CHECK(dataset.getDataType() == H5::PredType::NATIVE_FLOAT);

  auto num_ele = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<hsize_t>());
  std::vector<float> weights(num_ele);
  dataset.read(weights.data(), dataset.getDataType());

  dataspace.close();
  dataset.close();
  group.close();

  return {std::move(weights), dims};
}

void H5ModelReader::ExtractModelInfos(const json& config) {
  const auto& layers = config["layers"];

  for (auto& layer : layers) {
    auto name = layer["name"];
    name_to_layers_[name] = Layer(layer);
  }

  const auto& input_names = config["input_layers"];
  for (auto& input : input_names) {
    if (input[0].is_string()) {
      inputs_.push_back(&name_to_layers_[input[0]]);
    }
  }

  const auto& output_names = config["output_layers"];
  for (auto& output : output_names) {
    if (output[0].is_string()) {
      outputs_.push_back(&name_to_layers_[output[0]]);
    }
  }
}

json ModelPreprocessor::OptimizeConfig(const json& config) {
  json new_config = config;

  // 缓存所有 层级 到 Map，便于查找
  std::unordered_map<std::string, json*> layer_map;
  for (auto& layer : new_config["layers"]) {
    layer_map[layer["name"]] = &layer;
  }

  // 检测是否有集成度较高的层级, 若有则进行拆解。
  std::vector<json> new_layers;
  auto& output_layers = new_config["output_layers"];
  for (auto& output : output_layers) {
    if (output[0].is_string()) {
      auto output_name = output[0];
      auto new_layer_name = UnpackInlineLayer(layer_map, layer_map[output_name], new_layers);
      if (!new_layer_name.empty()) {
        output[0] = new_layer_name;
      }
    }
  }

  // 将分离出来的新层加入到 json 中
  for (auto& new_layer : new_layers) {
    new_config["layers"].push_back(new_layer);
  }
  return new_config;
}

std::string ModelPreprocessor::UnpackInlineLayer(std::unordered_map<std::string, json*>& layer_map,
                                                 json* layer, std::vector<json>& new_layers) {
  // 若节点被处理过，则直接返回
  if (layer->find("masked") != layer->end()) {
    return "";
  }

  (*layer)["masked"] = true;

  auto& inputs = layer->at("inbound_nodes")[0];

  // 递归处理上层节点
  for (auto& input : inputs) {
    auto name = input[0].get<std::string>();
    auto new_layer_name = UnpackInlineLayer(layer_map, layer_map[name], new_layers);
    if (!new_layer_name.empty()) {
      input[0] = new_layer_name;
    }
  }

  // 处理当层节点：查找当前层级是否存在被集成的子类层级，若存在，则抽出信息创建子类层级
  for (auto& type : inlined_layer_types_) {
    auto inline_layer = layer->at("config").find(type);
    if (inline_layer != layer->at("config").end()) {
      auto new_layer = CreateLayerByType(type, inline_layer.value(), layer);
      if (new_layer.empty()) return {};

      const std::string layer_name = new_layer.at("name");
      new_layers.push_back(std::move(new_layer));

      return layer_name;
    }
  }

  return "";
}

json ModelPreprocessor::CreateActivationJson(const std::string& type, json* parent) {
  if (parent->at("class_name").get<std::string>() == "Activation" ||
      parent->at("class_name").get<std::string>() == "SimpleRNN" ||
      parent->at("class_name").get<std::string>() == "LSTM" ||
      parent->at("class_name").get<std::string>() == "GRU" || type == "linear") {
    return {};
  }

  json activation;
  activation["name"] = parent->at("name").get<std::string>() + "_activation";
  activation["class_name"] = "Activation";

  json config;
  config["activation"] = type;
  activation["config"] = config;

  activation["inbound_nodes"] =
      std::vector<std::vector<std::vector<std::string>>>{{{parent->at("name")}}};

  return activation;
}

FWD_KERAS_NAMESPACE_END
