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
#include <string>
#include <unordered_map>
#include <vector>

#include "common/fwd_common.h"
#include "fwd_torch/fwd_torch_renaming.h"
#include "torch_desc_creators/i_torch_layer_creator.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief Torch 模型处理类
 */
class TorchModule {
 public:
  /**
   * \brief 构造器
   */
  TorchModule() = default;
  /**
   * \brief 析构器
   */
  ~TorchModule() = default;

  /**
   * \brief 加载模型
   * \param module_path 模型路径
   * \param mode 推理模式
   * \return 加载是否成功
   */
  bool Load(const std::string& module_path, InferMode mode);

  /**
   * \brief 利用伪输入对 Module 进行推断，从而获取所有中间节点的参数信息并缓存
   * \param inputs 伪输入
   * \return 成功则返回 true
   */
  bool EvalAll(const std::vector<c10::IValue>& inputs);

  /**
   * \brief 根据 torch jit value 得到对应的 IValue
   * \param value 输入 value
   * \return 对应的 IValue
   */
  const c10::IValue& Get(const JitValue* value) const;

  void Dump() const;

  /**
   * \brief 清楚上下文缓存
   */
  void Clear();

  /**
   * \brief 设置最大批量大小
   * \param size
   */
  void SetMaxBatchSize(int size);

  InferMode GetMode() const;

  int GetMaxBatchSize() const;

  const std::set<int>& GetUnusedInputSet() const;

  const std::vector<const JitValue*>& Inputs() const;

  at::ArrayRef<JitValue*> Outputs() const;

 private:
  /**
   * \brief Graph 预处理（在 EvalAll 之前），此类修改不会影响后续的 EvalAll
   * 操作。
   */
  void PreProcessGraph();

  /**
   * \brief Graph 后处理（在 EvalAll 之后），此类修改之后则无法再执行 EvalAll。
   */
  void PostProcessGraph();

  /**
   * \brief 拆解 Graph 内的 Tuple/List 类型输入，直至所有 Graph 输入类型 变为
   * TensorType。并移除无用的节点。
   */
  void RemoveUnusedInputs();

  /**
   * \brief 最大批量大小
   */
  int max_batch_size_{-1};

  /**
   * \brief 推理模式
   */
  InferMode mode_{InferMode::FLOAT};

  /**
   * \brief 未被使用的输入序号
   */
  std::set<int> unused_input_indices_;

  /**
   * \brief 有效输入节点
   */
  std::vector<const JitValue*> valid_inputs_;

  /**
   * \brief 有效输入节点映射
   */
  std::unordered_map<std::string, const JitValue*> input_map_;

  /**
   * \brief torch 模型
   */
  torch::jit::script::Module module_;

  /**
   * \brief torch graph
   */
  std::shared_ptr<torch::jit::Graph> graph_;

  /**
   * \brief 用于缓存中间节点参数信息的上下文
   */
  std::unordered_map<const JitValue*, c10::IValue> context_;
};

FWD_TORCH_NAMESPACE_END
