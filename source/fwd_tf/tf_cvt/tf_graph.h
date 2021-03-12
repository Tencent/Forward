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

#include <easylogging++.h>
#include <tensorflow/c/c_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN
enum class InferMode;

FWD_NAMESPACE_END

FWD_TF_NAMESPACE_BEGIN
class Operation;

class Graph {
 public:
  Graph() : graph_{TF_NewGraph()} {}

  ~Graph() { TF_DeleteGraph(graph_); }

  Graph(const Graph&) = delete;

  Graph& operator=(const Graph&) = delete;

  TF_Graph* get() const { return graph_; }

  bool Load(const std::string& graph_path, InferMode mode);

  InferMode Mode() const { return mode_; }

  TF_Operation* OperationByName(const std::string& oper_name) const {
    return TF_GraphOperationByName(graph_, oper_name.c_str());
  }

  const std::vector<Operation>& Inputs() const { return inputs_; }

  const std::vector<Operation>& Outputs() const { return outputs_; }

 private:
  /**
   * \brief 从 TF_Graph 提取 输入 和 输出 信息
   */
  bool ExtractGraphInfos();

  std::vector<Operation> AllOperations() const;

  /**
   * \brief 从外部文件加载权重，文件名与权重名相同
   * \param name 权重名
   * \param weights 待读取的权重缓存
   * \return 加载是否成功：成功返回 True
   */
  bool LoadWeightsFromFile(const std::string& name, std::vector<float>& weights) const;

  bool LoadOuterWeightsToOp(const Operation& op);

  TF_Graph* graph_{nullptr};

  InferMode mode_;

  std::vector<Operation> inputs_;

  std::vector<Operation> outputs_;

  /**
   * \brief 外部权重文件的根路径
   */
  std::string outer_root_path_{""};

  /**
   * \brief 外部权重映射
   */
  std::unordered_map<std::string, std::shared_ptr<TF_Tensor>> outer_weights_map_;
};

FWD_TF_NAMESPACE_END
