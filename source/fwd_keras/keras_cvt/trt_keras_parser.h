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

#include "common/common_macros.h"
#include "common/trt_network_desc.h"
#include "fwd_keras/keras_cvt/keras_cpp_api.h"
#include "keras_desc_creators/i_keras_layer_creator.h"

FWD_KERAS_NAMESPACE_BEGIN

/**
 * \brief 将 TF_Graph 转换为 网络层描述 的转换器
 */
class Parser {
 public:
  /**
   * \brief 构造器
   * \param data_type TensorRT 网络输入类型: kFLOAT or kHALF
   */
  explicit Parser(InferMode mode);

  /**
   * \brief 将 TF_Graph 转换为 网络层描述集合
   * \param graph_path Graph 文件路径
   * \param batch_size 批量大小
   * \return
   */
  bool Parse(const std::string& graph_path, int batch_size);

  /**
   * \brief 获取 网络层描述
   * \return
   */
  const TrtNetworkDesc& GetNetwork() const;

 private:
  /**
   * \brief 根据 batch_size 创建 InputDescs
   * \param inputs
   * \param batch_size
   */
  bool CreateInputDescs(const std::vector<const Layer*>& inputs, int batch_size);

  /**
   * \brief 对 输入层描述 进行 后处理：调整维度顺序
   * \param input
   * \param input_desc
   */
  void ProcessInputDesc(const Layer& input, std::shared_ptr<TrtInputDesc> input_desc);

  /**
   * \brief 对 输出层描述 进行 后处理：调整维度顺序
   * \param output_layer
   * \param output_desc
   */
  void ProcessOutputDesc(const Layer& output_layer, TrtOutputDesc* output_desc);

  /**
   * \brief 将 Graph 节点 转换为 网络层描述
   * \param parent TrtLayerDesc
   * \param layer TF_Operation
   * \return bool 转换是否成功
   */
  bool ParseOperaion(TrtLayerDesc* parent, const Layer& layer);

  /**
   * \brief 遍历查找 TF_Operation 对应的层描述器
   * \param layer TF_Operation 操作
   * \return Op 对应的层描述器；若未找到，则返回 nullptr。
   */
  ILayerDescCreator* FindDescCreator(const Layer& layer);

  /**
   * \brief 创建 TrtShuffleDesc 用于将 NHWC 格式转换为 NCHW 格式
   * \return shuffle 层描述，调用者需要给此描述的输入描述赋值
   */
  static std::shared_ptr<TrtShuffleDesc> CreateNHWC2NCHWLayerDesc();

  /**
   * \brief 创建 TrtShuffleDesc 用于将 NHWC 格式转换为 NCHW 格式
   * \return shuffle 层描述，调用者需要给此描述的输入描述赋值
   */
  static std::shared_ptr<TrtShuffleDesc> CreateNCHW2NHWCLayerDesc();

  /**
   * \brief 注册 网络层描述 创建器
   * \tparam T 网络层描述类型
   */
  template <typename T>
  void RegisterCreator() {
    using KerasLayerDescCreator = TLayerDescCreator<T>;
    layer_desc_creators_.push_back(std::make_shared<KerasLayerDescCreator>());
  }

  /**
   * \brief 已注册的 网络层描述 创建器
   */
  std::vector<std::shared_ptr<ILayerDescCreator>> layer_desc_creators_;

  /**
   * \brief 从 TF_Operation 到 网络层描述 的映射
   */
  std::unordered_map<const Layer*, std::shared_ptr<TrtLayerDesc>> value_to_layer_descs_;

  /**
   * \brief TensorRT 的推理模式
   */
  InferMode mode_;

  /**
   * \brief 网络层描述
   */
  TrtNetworkDesc network_;

  /**
   * \brief H5 模型读取器
   */
  H5ModelReader model_reader_;
};

FWD_KERAS_NAMESPACE_END
