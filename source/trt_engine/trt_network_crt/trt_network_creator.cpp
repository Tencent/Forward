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

#include "trt_engine/trt_network_crt/trt_network_creator.h"

FWD_TRT_NAMESPACE_BEGIN

TrtNetworkCreator::TrtNetworkCreator(nvinfer1::INetworkDefinition* network) : network_(network) {}

bool TrtNetworkCreator::Create(const TrtNetworkDesc& network_desc) {
  // create input layers
  for (const auto& input_desc : network_desc.inputs) {
    auto input = CreateLayer(input_desc.get());
    if (input.empty()) return false;
  }

  // create recursively layers from output to input
  for (const auto& output_desc : network_desc.outputs) {
    auto output = CreateLayer(output_desc.get());
    for (int i = 0; i < output.size(); ++i) {
      const auto& name =
          output[i]->getName() + std::string("::") + output_desc->name + "::" + std::to_string(i);
      output[i]->setName(name.c_str());
    }
    if (output.empty()) return false;
  }

  return true;
}

ITensorVector TrtNetworkCreator::CreateLayer(const TrtLayerDesc* layer_desc) {
  // if it has been created, return directly the output
  const auto iter = created_layers_.find(layer_desc);
  if (iter != created_layers_.end()) return iter->second;

  // find creator by name
  auto layer_creator = creator_manager_.FindCreator(layer_desc->Name());
  if (layer_creator == nullptr) return {};

  const auto layer_inputs = GetLayerInputs(layer_desc->inputs);
  auto layer_outputs = layer_creator->CreateLayer(network_, layer_desc, layer_inputs);

  // mark as created
  created_layers_[layer_desc] = layer_outputs;
  return layer_outputs;
}

ITensorVector TrtNetworkCreator::GetLayerInputs(const std::vector<TrtLayerOutput>& layer_inputs) {
  ITensorVector input_tensors;

  for (const auto& layer_input : layer_inputs) {
    auto& input_layer_desc = layer_input.layer_desc;
    if (!input_layer_desc) continue;

    // 对于 Noop，直接将多个输入添加到当前的输入
    if (input_layer_desc->Name() == TrtNoopDesc::NAME()) {
      const auto tensors = GetLayerInputs(input_layer_desc->inputs);
      input_tensors.insert(input_tensors.end(), tensors.begin(), tensors.end());
    } else {
      ITensorVector tensors = CreateLayer(input_layer_desc.get());
      if (tensors.empty()) return {};

      input_tensors.push_back(tensors[layer_input.index]);
    }
  }

  return input_tensors;
}

FWD_TRT_NAMESPACE_END
