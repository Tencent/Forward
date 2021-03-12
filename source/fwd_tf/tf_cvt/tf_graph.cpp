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

#include "fwd_tf/tf_cvt/tf_graph.h"

#include <memory>
#include <string>
#include <vector>

#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN

bool Graph::Load(const std::string& graph_path, InferMode mode) {
  if (graph_path.empty()) {
    return false;
  }

  const std::shared_ptr<TF_Buffer> buffer(Utils::ReadBufferFromFile(graph_path), TF_DeleteBuffer);
  if (buffer == nullptr) {
    return false;
  }

  const std::shared_ptr<TF_ImportGraphDefOptions> opts(TF_NewImportGraphDefOptions(),
                                                       TF_DeleteImportGraphDefOptions);
  Status status;
  TF_GraphImportGraphDef(graph_, buffer.get(), opts.get(), status);
  if (!status.Ok()) {
    return false;
  }

  outer_root_path_ = graph_path.substr(0, graph_path.find_last_of('/')) + "/";
  mode_ = mode;
  // 遍历一遍 Graph， 标记 输入 和 输出 节点
  return ExtractGraphInfos();
}

bool Graph::ExtractGraphInfos() {
  const auto ops = AllOperations();
  for (const auto& op : ops) {
    const std::string type = op.OpType();

    if (type == "Placeholder") {
      inputs_.push_back(op);
    } else if (type == "VariableV2") {
      LoadOuterWeightsToOp(op);
    } else if (type == "Const" || type == "Assert") {
      // skip to avoid to be tagged as output
      continue;
    }

    // TODO(yzx): 这里假设 TF_Operation 没有被其他节点使用则是输出
    if (op.OutputNumConsumers(0) == 0) {
      int is_output = 1;
      for (int i = 0; i < op.NumInputs(); ++i) {
        is_output &= (op.Input(i).OpType() != "IteratorGetNext");
      }

      if (is_output) outputs_.push_back(op);
    }
  }
  return true;
}

std::vector<Operation> Graph::AllOperations() const {
  std::vector<Operation> ops;
  size_t pos = 0;
  while (true) {
    auto* op = TF_GraphNextOperation(graph_, &pos);
    if (op == nullptr) {
      break;
    }
    ops.emplace_back(Operation(graph_, op));
  }
  return ops;
}

bool Graph::LoadWeightsFromFile(const std::string& name, std::vector<float>& weights) const {
  std::string file_name = name;  // .substr(0, name.find_last_of('/'));
  std::transform(file_name.begin(), file_name.end(), file_name.begin(),
                 [](char c) { return c == '/' ? '-' : c; });
  std::ifstream file(outer_root_path_ + file_name + ".w", std::ios::binary);

  if (!file.is_open()) {
    LOG(ERROR) << "Open file failed : " << outer_root_path_ + file_name;
    return false;
  }

  file.seekg(0, std::ios::end);
  const auto byte_size = file.tellg();
  file.seekg(0, std::ios::beg);

  weights.resize(byte_size / sizeof(float));

  file.read(reinterpret_cast<char*>(weights.data()), byte_size);

  file.close();

  return true;
}

bool Graph::LoadOuterWeightsToOp(const Operation& op) {
  const std::string key = op.Name();
  std::vector<float> weights;

  if (!LoadWeightsFromFile(key, weights)) {
    LOG(ERROR) << "Load Outer Weights failed.";
    return false;
  }

  const auto new_op_name = key + "_weights";
  TF_OperationDescription* new_op = TF_NewOperation(graph_, "Const", new_op_name.c_str());

  auto tensor =
      Utils::CreateTensor(TF_DataType::TF_FLOAT, op.GetAttrShape("shape"), weights.data());

  Status status;
  TF_SetAttrType(new_op, "dtype", TF_DataType::TF_FLOAT);
  TF_SetAttrTensor(new_op, "value", tensor.get(), status);

  TF_FinishOperation(new_op, status);
  if (!status.Ok()) {
    LOG(ERROR) << "Error msg : " << status.Message();
    return false;
  }
  outer_weights_map_[new_op_name] = tensor;
  return true;
}

FWD_TF_NAMESPACE_END
