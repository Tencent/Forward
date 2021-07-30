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

#include "fwd_torch/torch_cvt/torch_module_parser.h"

#include <memory>
#include <string>
#include <vector>

#include "common/fwd_common.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

Parser::Parser(InferMode mode) : mode_(mode) {}

bool Parser::Parse(const std::string& module_path, const std::vector<torch::jit::IValue>& inputs) {
  if (!module_.Load(module_path, mode_)) {
    LOG(ERROR) << "Load module failed, from :" << module_path;
    return false;
  }

  return CreateDescs(inputs);
}

bool Parser::Parse(const std::string& module_path,
                   const std::unordered_map<std::string, c10::IValue>& input_map) {
  if (!module_.Load(module_path, mode_)) {
    LOG(ERROR) << "Load module failed, from :" << module_path;
    return false;
  }

  std::vector<c10::IValue> inputs;
  for (auto& input : module_.Inputs()) {
    auto entry = input_map.find(input->debugNameBase());
    if (entry != input_map.end()) {
      inputs.push_back(entry->second);
    }
  }

  if (inputs.size() != input_map.size()) {
    LOG(ERROR) << "Invalid Input : Number of input is not matched with graph inputs";
    return false;
  }

  return CreateDescs(inputs);
}

bool Parser::CreateDescs(const std::vector<c10::IValue>& inputs) {
  std::vector<c10::IValue> f_inputs(inputs);
  if (!RegularizeIValues(f_inputs)) return false;

  if (!module_.EvalAll(f_inputs)) {
    LOG(ERROR) << "TorchModule EvalAll failed: invalid graph, or invalid "
                  "inputs, or module is not on cpu.";
    return false;
  }

  if (!CreateInputDescs(f_inputs)) return false;

  if (!SetNetworkBatchSize()) return false;

  const auto* graph_output = module_.Outputs()[0];
  const auto output_desc = std::make_shared<TrtOutputDesc>();
  if (!ParseValue(output_desc.get(), graph_output)) {
    return false;
  }

#ifdef ENABLE_TORCH_PLUGIN
  FuseTorchSubmodule(output_desc.get());
#endif

  // torch single output
  network_.outputs.push_back(output_desc);

  module_.Clear();

  return true;
}

bool Parser::ParseValue(TrtLayerDesc* parent, const JitValue* value) {
  // To support multi-input of JitNode, we use JitValue as the key for mapping TrtLayerDescs
  const auto* node = value->node();
  const auto& outputs = node->outputs();
  int value_idx = std::find(outputs.begin(), outputs.end(), value) - outputs.begin();
  CHECK_GE(value_idx, 0);

  // If it is created, it is directly added into parent->inputs
  const auto iter = created_desc_map_.find(value);
  if (iter != created_desc_map_.end()) {
    if (iter->second->Name() == TrtInputDesc::NAME()) {
      value_idx = 0;
    }
    parent->inputs.push_back({iter->second, value_idx});
    return true;
  }

  auto layer_creator = desc_manager_.FindDescCreator(node, module_);
  if (layer_creator == nullptr) return false;

  std::vector<const JitValue*> input_values;
  const auto layer_desc = layer_creator->Create(node, module_, input_values);

  if (layer_desc == nullptr || input_values.empty()) {
    LOG(ERROR) << "Creating " << node->kind().toQualString()
               << " Desc failed! Please Check implementation and inputs.";
    return false;
  }

  if (network_.torch_module_path.empty() && layer_desc->Name() == TrtTorchModuleDesc::NAME()) {
    network_.torch_module_path = module_.ModulePath();
  }

  // If the output is TensorList, we need to add all outputs in the TensorList into parent->inputs.
  // So far, this method is only designed for aten::split
  if (parent->Name() == TrtIdentityDesc::NAME() &&
      value->type()->kind() == c10::TypeKind::ListType) {
    const auto& output_value = module_.Get(value);
    if (output_value.isTensorList()) {
      for (size_t i = 0; i < output_value.toTensorList().size(); ++i) {
        parent->inputs.push_back({layer_desc, static_cast<int>(i)});
      }
    }
  } else {
    parent->inputs.push_back({layer_desc, value_idx});
  }

  // For multi-output nodes, we only create single TrtLayerDesc for it.
  for (auto output : outputs) {
    created_desc_map_[output] = layer_desc;
  }

  for (const auto input_value : input_values) {
    if (input_value != nullptr && !ParseValue(layer_desc.get(), input_value)) {
      return false;
    }
  }

  return true;
}

bool Parser::SetNetworkBatchSize() {
  const int batch_size = network_.inputs[0]->dimensions.d[0];
  for (auto& input : network_.inputs) {
    if (batch_size != input->dimensions.d[0]) {
      LOG(ERROR) << "Batch sizes of inputs are not consistent! Please check "
                    "inputs' dimensions.";
      return false;
    }

#ifdef USE_DYNAMIC_BATCH
    input->dimensions.d[0] = -1;
#endif  // USE_DYNAMIC_BATCH
  }

  // set the batch_size of network
  network_.batch_size = batch_size;
  module_.SetMaxBatchSize(network_.batch_size);

  return true;
}

bool Parser::CreateInputDescs(const std::vector<c10::IValue>& inputs) {
  // create input desc from dummy_inputs
  const auto& g_inputs = module_.Inputs();

  if (g_inputs.size() != inputs.size()) {
    LOG(ERROR) << "The number of inputs is not matched with the number of "
                  "graph inputs";
    return false;
  }

  for (int i = 0; i < g_inputs.size(); ++i) {
    const auto& g_input = g_inputs[i];
    const auto name = g_input->debugNameBase();
    const auto unpacked_inputs = UnpackIValues({inputs[i]});
    const auto unpakced_g_inputs = UnpackJitValue(g_input);

    if (unpakced_g_inputs.size() != unpacked_inputs.size()) {
      LOG(ERROR) << "The number of inputs is not matched with the number of "
                    "graph inputs";
      return false;
    }

    for (int j = 0; j < unpacked_inputs.size(); ++j) {
      const auto input = unpacked_inputs[j].toTensor();

      auto input_desc = std::make_shared<TrtInputDesc>();
      input_desc->name = name + std::to_string(j);
      if (!SetInputType(input_desc, input.scalar_type())) return false;
      input_desc->dimensions = DimsOf(input);

      created_desc_map_[unpakced_g_inputs[j]] = input_desc;
      network_.inputs.push_back(input_desc);
    }
  }

  return true;
}

bool Parser::SetInputType(std::shared_ptr<TrtInputDesc> input_desc,
                          const c10::ScalarType& input_type) const {
  switch (input_type) {
    case c10::kFloat:
    case c10::kHalf:
      input_desc->type =
          mode_ == InferMode::FLOAT ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
      break;
    case c10::kLong:
    case c10::kInt:
      input_desc->type = nvinfer1::DataType::kINT32;
      break;
    default:
      LOG(ERROR) << "Unsupported Input type for Input = ";
      return false;
  }
  return true;
}

bool Parser::FuseTorchSubmodule(TrtLayerDesc* current) {
  // End condition
  if (current->Name() == TrtInputDesc::NAME()) return false;

  int need_fuse = 1;
  for (auto& input : current->inputs) need_fuse &= FuseTorchSubmodule(input.layer_desc.get());

  if (current->Name() == TrtTorchModuleDesc::NAME()) {
    // fuse nodes
    if (need_fuse) {
      std::vector<TrtLayerOutput> inputs;
      std::vector<int> node_ids;
      std::vector<int> in_types;

      // collect fuse infos
      for (auto& input : current->inputs) {
        auto child = reinterpret_cast<TrtTorchModuleDesc*>(input.layer_desc.get());
        node_ids.insert(node_ids.end(), child->node_ids.begin(), child->node_ids.end());
        in_types.insert(in_types.end(), child->in_types.begin(), child->in_types.end());
        inputs.insert(inputs.end(), child->inputs.begin(), child->inputs.end());
      }

      // update
      auto* cur_desc = reinterpret_cast<TrtTorchModuleDesc*>(current);
      cur_desc->inputs.swap(inputs);
      cur_desc->in_types.swap(in_types);
      node_ids.insert(node_ids.end(), cur_desc->node_ids.begin(), cur_desc->node_ids.end());
      cur_desc->node_ids.swap(node_ids);
    }

    return true;
  }

  return false;
}
FWD_TORCH_NAMESPACE_END
