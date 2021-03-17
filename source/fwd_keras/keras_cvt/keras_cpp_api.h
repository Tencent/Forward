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

#include <H5Cpp.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "common/common_macros.h"
#include "common/fwd_weights.h"
#include "common/trt_utils.h"

#ifdef _MSC_VER
#undef max
#undef min
#endif

FWD_KERAS_NAMESPACE_BEGIN
using json = nlohmann::json;

/**
 * \brief 辅助工具类
 */
inline nvinfer1::Dims DimsOf(const json& dims_json) {
  nvinfer1::Dims dims;
  dims.nbDims = dims_json.size();
  for (int i = 0; i < dims.nbDims; ++i) {
    const auto item = dims_json[i];
    // std::cout << item.type_name() <<std::endl;
    dims.d[i] = item == nullptr ? -1 : item.get<int>();
  }

  return dims;
}

class Layer {
 public:
  Layer() = default;

  explicit Layer(const json& layer) : layer_(layer) {}

  ~Layer() = default;

  const Layer* get() const { return this; }

  std::string Name() const { return layer_.at("name"); }

  std::string Type() const { return layer_.at("class_name"); }

  std::vector<std::string> Inputs() const {
    std::vector<std::string> input_names;
    const auto& inputs_json = layer_.at("inbound_nodes").at(0);
    for (auto& input : inputs_json) {
      input_names.push_back(input.at(0));
    }
    return input_names;
  }

  template <typename DataType>
  DataType GetAttr(const std::string& name) const {
    // TODO(Ao Li): 这里转化失败会有异常抛出，考虑怎么处理
    return layer_.at("config").at(name).get<DataType>();
  }

  bool HasAttr(const std::string& name) const { return layer_.contains(name); }

 private:
  json layer_;
};

/**
 * \brief Keras 参数类
 */
class KerasWeights {
 public:
  KerasWeights() {}

  KerasWeights(const std::vector<float>& data, const std::vector<hsize_t>& dims) {
    data_ = data;
    dims_ = TrtUtils::ToDims(dims);
  }

  KerasWeights(std::vector<float>&& data, const std::vector<hsize_t>& dims) {
    data_ = std::move(data);
    dims_ = TrtUtils::ToDims(dims);
  }

  nvinfer1::Dims GetDimension() const { return dims_; }

  KerasWeights GetRow(int i) {
    CHECK_EQ(dims_.nbDims, 2);
    const int stride = dims_.d[1];
    const std::vector<float> data(data_.data() + i * stride, data_.data() + (i + 1) * stride);
    return {data, {static_cast<hsize_t>(stride)}};
  }

  const std::vector<float>& GetData() const { return data_; }

  size_t GetCount() const { return data_.size(); }

  FwdWeights ToFwdWeights() const { return FwdWeights(data_); }

 private:
  std::vector<float> data_;

  nvinfer1::Dims dims_{};
};

/**
 * \brief H5 模型预处理器
 */
class ModelPreprocessor {
 public:
  /**
   * \brief 优化模型配置，以便于解析模型
   * \param config 模型配置
   * \return 优化后的模型配置
   */
  json OptimizeConfig(const json& config);

  /**
   * \brief 从集成度较高的层级中，分离出一些支持层级，如 Activation 等。
   * \param layer_map
   * \param layer
   * \param new_layers
   * \return
   */
  std::string UnpackInlineLayer(std::unordered_map<std::string, json*>& layer_map, json* layer,
                                std::vector<json>& new_layers);

  /**
   * \brief 根据层级类型，创建层级
   * \tparam Ts 层级构建所需参数类型
   * \param layer_type 层级类型
   * \param params 层级构建所需参数
   * \return 新的层级
   */
  template <typename... Ts>
  json CreateLayerByType(const std::string& layer_type, Ts&&... params) {
    // TODO(yzx): 目前仅实现了 activation 层的 unpack
    if (layer_type == "activation") {
      return CreateActivationJson(std::forward<Ts>(params)...);
    }

    return {};
  }

  /**
   * \brief 根据所给参数，创建 Activation 层
   * \param type Activation Type
   * \param parent 父节点
   * \return
   */
  json CreateActivationJson(const std::string& type, json* parent);

 private:
  const std::vector<std::string> inlined_layer_types_{"activation"};
};

/**
 * \brief Keras HDF5 模型加载器
 */
class H5ModelReader {
 public:
  H5ModelReader() = default;

  ~H5ModelReader() {
    model_weights_.close();
    file_.close();
  }

  H5ModelReader(const H5ModelReader& other) = delete;

  H5ModelReader(H5ModelReader&& other) noexcept = delete;

  H5ModelReader& operator=(const H5ModelReader& other) = delete;

  H5ModelReader& operator=(H5ModelReader&& other) noexcept = delete;

  /**
   * \brief 模型输入
   * \return
   */
  const std::vector<const Layer*>& Inputs() { return inputs_; }

  /**
   * \brief 模型输出
   * \return
   */
  const std::vector<const Layer*>& Outputs() { return outputs_; }

  const Layer& GetLayer(const std::string& name) { return name_to_layers_[name]; }

  /**
   * \brief 加载模型
   * \param model_path 模型路径
   * \return 成功，返回True；
   */
  bool LoadModel(const std::string& model_path);

  /**
   * \brief 读取权重
   * \param group_name
   * \param weight_name
   * \return
   */
  KerasWeights ReadWeight(const std::string& group_name, const std::string& weight_name) const;

 private:
  /**
   * \brief 加载模型配置文件
   * \return
   */
  json ReadModelConfig() const;

  /**
   * \brief 抽取模型信息
   */
  void ExtractModelInfos(const json& config);

  H5::H5File file_;

  H5::Group model_weights_;

  /**
   * \brief H5 模型预处理器
   */
  ModelPreprocessor model_preprocessor_;

  /**
   * \brief 层级 Name -> Json 的映射
   */
  std::unordered_map<std::string, Layer> name_to_layers_;

  /**
   * \brief 输入层级
   */
  std::vector<const Layer*> inputs_;

  /**
   * \brief 输出层级
   */
  std::vector<const Layer*> outputs_;
};

FWD_KERAS_NAMESPACE_END
