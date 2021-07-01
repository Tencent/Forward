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

FWD_TORCH_NAMESPACE_BEGIN

/// When some torch nodes are not supported by Forward, we utilize TorchSubModule to create a
/// torch_module_plugin for TRT engine.
///
/// Some member functions are referred to TRTorch(https://github.com/NVIDIA/TRTorch): CloneNode,
/// GetOrAddInputForValue, RegisterOutputs
/// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
class TorchSubModule {
 public:
  TorchSubModule();
  ~TorchSubModule() = default;

  /// Add graph nodes into sub_module, the nodes will be cloned from original graph to a new graph.
  /// The meta data of nodes will also be cloned.
  bool AddNode(torch::jit::Node* node);

  bool AddNodes(const std::vector<torch::jit::Node*>& nodes);

  /// Based on added nodes in the new graph and added attributes, create a new TorchScript Module.
  bool CreateModule();

  /// Add attributes including parameters such that the new graph in the new module is able to
  /// prim::GetAttr from attributes.
  void CloneAttributesFrom(const torch::jit::Module& module);

  void ToCuda() { module_->to(c10::kCUDA); }

  /// Eval the created module with inputs to get results.
  /// The results will be unpacked to std::vector<c10::IValue>
  std::vector<torch::jit::IValue> Eval(const std::vector<torch::jit::IValue>& jit_inputs_ivalues);

  std::shared_ptr<torch::jit::Module> GetModule() { return module_; }

  const std::vector<torch::jit::Value*>& Inputs() const { return inputs_; }

 private:
  /// Clone nodes from the original graph to the new graph.
  torch::jit::Node* CloneNode(torch::jit::Node* node);

  /// Get/Add inputs(torch::jit::Value) of torch::jit::Node and clone it
  torch::jit::Value* GetOrAddInputForValue(torch::jit::Value* old_value);

  /// Register the new graph's outputs
  bool RegisterOutputs();

  std::shared_ptr<torch::jit::Graph> graph_;
  std::shared_ptr<torch::jit::Module> module_;

  std::vector<std::string> attr_names_;
  std::vector<torch::jit::Value*> inputs_;
  std::vector<torch::jit::Value*> outputs_;
  std::vector<torch::jit::Node*> nodes_;
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;
  std::unordered_map<std::string, c10::IValue> named_attrs_;
};

FWD_TORCH_NAMESPACE_END
