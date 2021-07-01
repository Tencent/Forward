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
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_submodule.h"

FWD_TORCH_NAMESPACE_BEGIN

static const std::unordered_map<c10::ScalarType, nvinfer1::DataType> TORCH2TRT_DTYPE_MAP{
    {c10::kFloat, nvinfer1::DataType::kFLOAT},
    {c10::kHalf, nvinfer1::DataType::kHALF},
    {c10::kInt, nvinfer1::DataType::kINT32},
    {c10::kLong, nvinfer1::DataType::kINT32},
};

// TorchModulePlugin 层描述创建器
template <>
class TLayerDescCreator<TrtTorchModuleDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    return node->kind().is_aten();
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TrtTorchModuleDesc::Create";

    const auto results = CreateAndEvalSubModule(node, module, input_values);

    // if no valid inputs, then create constant layer
    if (input_values.empty()) {
      input_values.push_back(nullptr);
      return CreateConstantLayer(results);
    }

    return CreateTorchModulePluginLayer(node, module, input_values, results);
  }

 private:
  std::vector<torch::jit::IValue> CreateAndEvalSubModule(
      const fwd::torch_::JitNode* node, const fwd::torch_::TorchModule& module,
      std::vector<const fwd::torch_::JitValue*>& input_values) {
    // Create SubModule
    TorchSubModule sub_module;
    CHECK(sub_module.AddNode(const_cast<torch::jit::Node*>(node)));
    sub_module.CloneAttributesFrom(module.GetModule());
    CHECK(sub_module.CreateModule());

    // set number of inputs
    const auto& inputs = sub_module.Inputs();

    std::vector<c10::IValue> dummy_inputs;
    for (int i = 0; i < inputs.size(); ++i) {
      input_values.push_back(inputs[i]);
      dummy_inputs.push_back(module.Get(inputs[i]));
    }

    // set output_dims and nb_outputs
    return sub_module.Eval(dummy_inputs);
  }

  const std::shared_ptr<fwd::TrtLayerDesc> CreateTorchModulePluginLayer(
      const fwd::torch_::JitNode* node, const fwd::torch_::TorchModule& module,
      const std::vector<const fwd::torch_::JitValue*>& input_values,
      const std::vector<torch::jit::IValue>& res) {
    auto layer_desc = std::make_shared<TrtTorchModuleDesc>();
    // now only support one node
    layer_desc->node_ids.push_back(node->outputs()[0]->unique());
    // now module path is fixed as the given module path
    layer_desc->module_path = module.ModulePath();

    // assign input meta infos
    for (auto& input : input_values) {
      const auto tensor = module.Get(input);
      CHECK(tensor.isTensor());
      layer_desc->in_types.push_back(static_cast<int>(tensor.toTensor().scalar_type()));
    }

    // assign output meta infos
    for (int i = 0; i < res.size(); ++i) {
      CHECK(res[i].isTensor());
      auto tensor = res[i].toTensor();
      auto dtype = TORCH2TRT_DTYPE_MAP.find(tensor.scalar_type());
      if (dtype == TORCH2TRT_DTYPE_MAP.end()) {
        LOG(ERROR) << "Output's data_type " << static_cast<int>(tensor.scalar_type())
                   << " is not supported.";
        return nullptr;
      }
      if (tensor.scalar_type() == c10::kLong) {
        LOG(WARNING) << "c10::kLong output will be automatically cast to c10::kInt";
      }
      layer_desc->out_types.push_back(dtype->second);
      layer_desc->out_dims.push_back(DimsOf(tensor));
    }

    return layer_desc;
  }

  const std::shared_ptr<fwd::TrtLayerDesc> CreateConstantLayer(
      const std::vector<torch::jit::IValue>& res) {
    // only support output 1 constant tensor
    CHECK(res.size() == 1 && res[0].isTensor());
    auto tensor = res[0].toTensor();
    auto const_layer_desc = std::make_shared<TrtConstantDesc>();
    const_layer_desc->weights = ToFwdWeights(tensor);
    const_layer_desc->dimensions = DimsOf(tensor);
    return const_layer_desc;
  }
};

FWD_TORCH_NAMESPACE_END
