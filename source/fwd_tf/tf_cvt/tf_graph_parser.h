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
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/trt_network_desc.h"
#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"
#include "fwd_tf/tf_cvt/tf_desc_manager.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_NAMESPACE_BEGIN
enum class InferMode;

FWD_NAMESPACE_END

FWD_TF_NAMESPACE_BEGIN

/**
 * \brief 将 TF_Graph 转换为 网络层描述 的转换器
 */
class Parser {
 public:
  /**
   * \brief 构造器
   * \param mode 推理模式
   */
  explicit Parser(InferMode mode);

  /**
   * \brief 将 TF_Graph 转换为 网络层描述集合
   * \param graph_path Graph 文件路径
   * \param dummy_input_map 批量大小
   * \return
   */
  bool Parse(const std::string& graph_path,
             const std::unordered_map<std::string, TF_Tensor*>& dummy_input_map);

  /**
   * \brief 获取 网络层描述
   * \return
   */
  const TrtNetworkDesc& GetNetwork() const;

  /**
   * \brief 获取无用输入的 Id
   * \return 无用输入的 Id 集合
   */
  std::set<int> GetUnusedInputs() const;

 private:
  /** 根据 dummy input 设置输入数据类型
   * \brief
   * \param input dummy input
   * \param input_desc
   * \param dummy_input
   * \return
   */
  bool SetInputType(const Operation& input, std::shared_ptr<TrtInputDesc> input_desc,
                    const TF_Tensor* dummy_input) const;

  /**
   * \brief 根据 batch_size 创建 InputDescs
   * \param inputs
   * \param dummy_input_map
   */
  bool CreateInputDescs(const std::vector<Operation>& inputs,
                        const std::unordered_map<std::string, TF_Tensor*>& dummy_input_map);

  /**
   * \brief 将 Graph 节点 转换为 网络层描述
   * \param parent TrtLayerDesc
   * \param op TF_Operation
   * \return bool 转换是否成功
   */
  bool ParseOperaion(TrtLayerDesc* parent, const Operation& op, int index);

  /**
   * \brief 对 输入层描述 进行 后处理：调整维度顺序
   * \param input
   * \param input_desc
   */
  void ProcessInputDesc(const Operation& input, std::shared_ptr<TrtInputDesc> input_desc);

  /**
   * \brief 对 输出层描述 进行 后处理：调整维度顺序
   * \param graph_output
   * \param output_desc
   */
  void ProcessOutputDesc(const Operation& graph_output, TrtOutputDesc* output_desc) const;

  /**
   * \brief 设置网络批量大小
   * \return
   */
  bool SetNetworkBatchSize();

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
   * \brief 从 TF_Operation 到 网络层描述 的映射
   */
  std::unordered_map<const TF_Operation*, std::shared_ptr<TrtLayerDesc>> created_desc_map_;

  /**
   * \brief TensorRT 的推理模式
   */
  InferMode mode_;

  /**
   * \brief 网络层描述
   */
  TrtNetworkDesc network_;

  /**
   * \brief Tf Graph
   */
  Graph graph_;

  /**
   * \brief TF 描述创建器管理器
   */
  TfDescManager desc_manager_;
};

FWD_TF_NAMESPACE_END
