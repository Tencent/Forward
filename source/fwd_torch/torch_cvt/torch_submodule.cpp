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

#include "fwd_torch/torch_cvt/torch_submodule.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

FWD_TORCH_NAMESPACE_BEGIN

inline bool isTensorOrTensorList(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get()) ||
         val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
}

/// Generate graph schema for a runable module.
/// This function are referred to TRTorch(https://github.com/NVIDIA/TRTorch)
/// Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
inline c10::FunctionSchema GenerateGraphSchema(std::string method_name,
                                               std::shared_ptr<torch::jit::Graph>& g) {
  std::vector<c10::Argument> args;
  for (auto in : g->inputs()) {
    args.push_back(c10::Argument(in->debugName(), in->type()));
  }

  std::vector<c10::Argument> returns;
  for (auto out : g->outputs()) {
    returns.push_back(c10::Argument(out->debugName(), out->type()));
  }

  return c10::FunctionSchema(method_name, method_name, args, returns);
}

TorchSubModule::TorchSubModule() {
  graph_ = std::make_shared<torch::jit::Graph>();
  module_ = std::make_shared<torch::jit::Module>(c10::QualifiedName("fwd_torch_sub_module"));
  auto self = graph_->addInput("self.1");
  self->setType(module_->type());
}

bool TorchSubModule::AddNode(torch::jit::Node* node) {
  nodes_.push_back(node);
  return CloneNode(node);
}

bool TorchSubModule::AddNodes(const std::vector<torch::jit::Node*>& nodes) {
  for (auto& node : nodes) {
    if (!AddNode(node)) return false;
  }
  return true;
}

torch::jit::Node* TorchSubModule::CloneNode(torch::jit::Node* node) {
  auto* block = graph_->block();

  auto env = [&](torch::jit::Value* v) { return GetOrAddInputForValue(v); };

  // create node for current graph by using the metadata in node and input Values in env
  auto new_node = block->appendNode(graph_->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new_[oo] = no;
  }
  return new_node;
}

torch::jit::Value* TorchSubModule::GetOrAddInputForValue(torch::jit::Value* old_value) {
  if (old_to_new_.count(old_value) == 0) {
    auto node = old_value->node();
    auto kind = node->kind();

    if (kind == c10::prim::Constant) {
      // create constants
      auto new_const = graph_->createClone(node, {nullptr});
      graph_->block()->prependNode(new_const);
      old_to_new_[old_value] = new_const->output();
      return new_const->output();
    } else if (kind == c10::prim::ListConstruct || kind == c10::prim::GetAttr ||
               kind == c10::prim::NumToTensor || kind == c10::aten::size ||
               kind == c10::aten::Int || kind == c10::aten::floor || kind == c10::aten::to ||
               kind == c10::aten::mul || kind == c10::aten::div) {
      // create ListConstruct constants and attribute params
      auto env = [&](torch::jit::Value* v) { return GetOrAddInputForValue(v); };
      auto new_const = graph_->createClone(node, env);
      graph_->block()->appendNode(new_const);
      old_to_new_[old_value] = new_const->output();
      return new_const->output();
    } else {
      torch::jit::Value* new_input;
      if (old_value->debugNameBase() == "self") {
        // use created 'self' input
        new_input = graph_->inputs()[0];
      } else {
        // every time when we addInput, we push back the corresponding lowering graph
        // torch::jit::Value to our raw_inputs
        new_input = graph_->block()->addInput();
        new_input->copyMetadata(old_value);
        inputs_.push_back(old_value);
      }
      old_to_new_[old_value] = new_input;
      return new_input;
    }
  }
  return old_to_new_[old_value];
}

bool TorchSubModule::CreateModule() {
  if (!RegisterOutputs()) return false;

  auto copy_g = graph_->copy();

  // create a module to run the graph
  for (auto& entry : named_attrs_) {
    module_->register_attribute(entry.first, entry.second.type(), entry.second);
  }

  auto cur_method = module_->_ivalue()->compilation_unit()->create_function(
      c10::QualifiedName("forward"), copy_g);
  auto schema = GenerateGraphSchema(cur_method->name(), copy_g);
  module_->type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  return true;
}

void TorchSubModule::AddAttributes(const torch::jit::named_attribute_list& attr_map) {
  for (auto attr : attr_map) named_attrs_[attr.name] = attr.value.deepcopy();
}

std::vector<torch::jit::IValue> TorchSubModule::Eval(
    const std::vector<torch::jit::IValue>& jit_inputs_ivalues) {
  // run segments to get outputs for later segments input shape, and other arguments such as Int
  std::vector<torch::jit::IValue> jit_results;

  torch::jit::IValue jit_results_ivalues = module_->forward(jit_inputs_ivalues);

  if (jit_results_ivalues.isTuple()) {
    auto results = jit_results_ivalues.toTuple()->elements();
    for (const auto& r : results) {
      jit_results.push_back(r);
    }
  } else if (jit_results_ivalues.isList()) {
    auto results = jit_results_ivalues.toList();
    for (const auto& r : results) {
      jit_results.push_back(r);
    }
  } else {
    jit_results.push_back(jit_results_ivalues);
  }

  return jit_results;
}

bool TorchSubModule::RegisterOutputs() {
  // find the corresponding raw values in original global graph for this segmented block's
  // inputs/outputs
  std::set<torch::jit::Value*> input_values;
  for (auto& input : inputs_) {
    input_values.insert(input);
  }

  for (auto& graph_output : graph_->outputs()) {
    input_values.insert(graph_output);
  }

  // should be careful here because some in-place operations don't return any values, there is no
  // output for this kind of segment identify the output for each mini-graph by checking if any
  // value in this graph is used later we shouldn't register nonTensor output for TensorRT segments
  for (auto& mini_graph_input : input_values) {
    if (std::find(inputs_.begin(), inputs_.end(), mini_graph_input) == inputs_.end() &&
        old_to_new_.count(mini_graph_input)) {
      if (!isTensorOrTensorList(mini_graph_input)) continue;
      outputs_.push_back(mini_graph_input);
      graph_->registerOutput(old_to_new_[mini_graph_input]);
    }
  }
  // if no output, then register the last node's output as current graph's output
  if (outputs_.empty()) {
    // register last nonInput Tensor outputs
    for (int i = nodes_.size() - 1; i >= 0; --i) {
      for (auto node_output : nodes_[i]->outputs()) {
        if (isTensorOrTensorList(node_output)) {
          outputs_.push_back(node_output);
          graph_->registerOutput(old_to_new_[node_output]);
        }
      }
      if (!outputs_.empty()) break;
    }
  }

  // merge multi-output to a tuple output such that graph can be forwardable.
  if (outputs_.size() > 1) {
    auto tuple_node = graph_->createTuple(graph_->outputs());
    auto tuple_output = graph_->appendNode(tuple_node);
    for (int i = graph_->outputs().size() - 1; i >= 0; --i) {
      graph_->eraseOutput(i);
    }
    graph_->registerOutput(tuple_output->output());
  }

  torch::jit::EliminateDeadCode(graph_);

  return !outputs_.empty();
}

FWD_TORCH_NAMESPACE_END
