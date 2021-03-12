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

#include <torch/script.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_macros.h"
#include "common/trt_network_desc.h"
#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_desc_manager.h"
#include "fwd_torch/torch_cvt/torch_module.h"

// 解决外部 log 与 torch log 冲突导致日志打不出的问题
#ifdef LOG
#undef LOG
#endif

#define LOG(LEVEL) CLOG(LEVEL, ELPP_CURR_FILE_LOGGER_ID)

#include <easylogging++.h>

FWD_NAMESPACE_BEGIN
enum class InferMode;

FWD_NAMESPACE_END

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief 将 Torch JIT Module 转换为 网络层描述 的转换器
 */
class Parser {
 public:
  /**
   * \brief 构造器
   * \param mode TensorRT 网络推理类型
   */
  explicit Parser(InferMode mode);

  /**
   * \brief 将 Torch JIT Module 转换为 网络层描述集合
   * \param module_path Torch JIT Module
   * \param inputs 伪输入
   * \return
   */
  bool Parse(const std::string& module_path, const std::vector<torch::jit::IValue>& inputs);

  /**
   * \brief 将 Torch JIT Module 转换为 网络层描述集合
   * \param module_path Torch JIT Module
   * \param input_map 伪输入映射
   * \return
   */
  bool Parse(const std::string& module_path,
             const std::unordered_map<std::string, c10::IValue>& input_map);

  /**
   * \brief 获取网络层描述
   * \return
   */
  const TrtNetworkDesc& GetNetwork() const { return network_; }

  /**
   * \brief 获取数据类型
   * \return
   */
  InferMode GetMode() const { return mode_; }

  /**
   * \brief 获取 Graph 中未使用的输入序号集合
   * \return
   */
  const std::set<int>& GetUnusedInputs() const { return network_.unused_input_indices; }

 private:
  /**
   * \brief 根据 dummy_input_map 解析 Torch  module
   * \param inputs
   * \return
   */
  bool CreateDescs(const std::vector<c10::IValue>& inputs);

  /**
   * \brief 根据 伪输入 创建 输入层描述
   * \param inputs 伪输入
   */
  bool CreateInputDescs(const std::vector<c10::IValue>& inputs);

  /**
   * \brief 将 Graph 节点 转换为 网络层描述
   * \param parent TrtLayerDesc
   * \param value JitValue
   * \return
   */
  bool ParseValue(TrtLayerDesc* parent, const JitValue* value);

  /**
   * \brief 设置输入类型
   * \param input_desc 输入描述
   * \param input_type torch  IValue 输入
   * \return
   */
  bool SetInputType(std::shared_ptr<TrtInputDesc> input_desc,
                    const c10::ScalarType& input_type) const;

  /**
   * \brief 校验并设置网络的批量大小
   * \return
   */
  bool SetNetworkBatchSize();

  /**
   * \brief TensorRT 的推理类型
   */
  InferMode mode_;

  /**
   * \brief 已创建过的从 Graph 节点 到 网络层描述 的映射
   */
  std::unordered_map<const JitValue*, std::shared_ptr<TrtLayerDesc>> created_desc_map_;

  /**
   * \brief 网络层描述
   */
  TrtNetworkDesc network_;

  /**
   * \brief Torch Module 管理器
   */
  TorchModule module_;

  /**
   * \brief Torch 描述管理器
   */
  TorchDescManager desc_manager_;
};

FWD_TORCH_NAMESPACE_END
