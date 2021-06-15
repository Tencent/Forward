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

#include "trt_engine/trt_network_crt/plugins/torch_module_plugin/torch_module_plugin.h"

#include <cuda_fp16.h>
#include <torch/script.h>
#include <functional>
#include <numeric>
#include <string>

#include "fwd_torch/torch_cvt/torch_helper.h"
#include "fwd_torch/torch_cvt/torch_module.h"
#include "fwd_torch/torch_cvt/torch_submodule.h"
#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

FWD_TRT_NAMESPACE_BEGIN

static const std::unordered_map<nvinfer1::DataType, c10::ScalarType> TRT2TORCH_DTYPE_MAP = {
    {nvinfer1::DataType::kFLOAT, c10::kFloat},
    {nvinfer1::DataType::kHALF, c10::kHalf},
    {nvinfer1::DataType::kINT32, c10::kInt},
    {nvinfer1::DataType::kINT8, c10::kChar},
};

TorchModulePlugin::TorchModulePlugin(nvinfer1::DataType data_type, const std::vector<int>& node_ids,
                                     const std::vector<int>& in_types,
                                     const std::vector<int>& out_types,
                                     const std::vector<nvinfer1::Dims>& out_dims)
    : data_type_(data_type),
      node_ids_(node_ids),
      in_types_(in_types),
      out_types_(out_types),
      out_dims_(out_dims) {
  ASSERT(!node_ids_.empty());
  ASSERT(!in_types_.empty());
  ASSERT(!out_types_.empty());
  ASSERT(!out_dims_.empty());
}

TorchModulePlugin::TorchModulePlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &node_ids_);
  deserialize_value(&serialData, &serialLength, &in_types_);
  deserialize_value(&serialData, &serialLength, &out_types_);
  deserialize_value(&serialData, &serialLength, &out_dims_);
}

TorchModulePlugin::~TorchModulePlugin() { terminate(); }

int TorchModulePlugin::getNbOutputs() const { return out_dims_.size(); }

nvinfer1::DimsExprs TorchModulePlugin::getOutputDimensions(int outputIndex,
                                                           const nvinfer1::DimsExprs* inputs,
                                                           int nbInputs,
                                                           nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs out_dim(inputs[0]);
  nvinfer1::Dims const_out_dim_ref = out_dims_[outputIndex];
  out_dim.nbDims = const_out_dim_ref.nbDims;
  // TODO(percyuan): so far, only support fixed output dimensions
  for (int i = 0; i < out_dim.nbDims; ++i) {
    out_dim.d[i] = exprBuilder.constant(const_out_dim_ref.d[i]);
  }
  return out_dim;
}

size_t TorchModulePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                           const nvinfer1::PluginTensorDesc* outputs,
                                           int nbOutputs) const {
  return 0;
}

int TorchModulePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                               const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) {
  auto options = c10::TensorOptions().layout(::torch::kStrided).requires_grad(false);
  // prepare CPU or GPU inputs
  std::vector<c10::IValue> torch_inputs;
  std::vector<int64_t> shape(inputDesc[0].dims.nbDims);
  for (int i = 0; i < in_types_.size(); ++i) {
    shape.assign(inputDesc[i].dims.d, inputDesc[i].dims.d + inputDesc[i].dims.nbDims);
    const auto dtype = TRT2TORCH_DTYPE_MAP.find(inputDesc[i].type);
    CHECK(dtype != TRT2TORCH_DTYPE_MAP.end());
    options = options.dtype(dtype->second);
    void* input = const_cast<void*>(inputs[i]);

#ifdef TORCH_HAS_CUDA
    options = options.device(c10::kCUDA);
#else
    options = options.device(c10::kCPU);
    const int ele_size = TrtCommon::GetElementSize(inputDesc[i].type);
    const auto volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    // copy data
    CUDA_CHECK(cudaMemcpyAsync(p_inputs_[i].Data(), input, volume * ele_size,
                               cudaMemcpyDeviceToHost, stream));
    input = p_inputs_[i].Data();
#endif  // TORCH_HAS_CUDA

    torch_inputs.push_back(torch::from_blob(input, shape, options));
  }

#ifndef TORCH_HAS_CUDA
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif  // TORCH_HAS_CUDA

  // support kLong inputs
  // TODO(percyyuan): maybe can try non-blocking=true
  for (int i = 0; i < torch_inputs.size(); ++i) {
    const auto real_dtype = static_cast<c10::ScalarType>(in_types_[i]);
    torch_inputs[i] = torch_inputs[i].toTensor().to(real_dtype);
  }

  // eval for results
  std::vector<c10::IValue> res = sub_module_->Eval(torch_inputs);

  // copy back results from tensor to ITensors
  for (int i = 0; i < res.size(); ++i) {
    torch::Tensor tensor = res[i].toTensor().contiguous();
    auto dtype = TRT2TORCH_DTYPE_MAP.find(outputDesc[i].type);
    CHECK(dtype != TRT2TORCH_DTYPE_MAP.end());
    tensor = tensor.to(dtype->second);
#ifdef TORCH_HAS_CUDA
    tensor = tensor.to(c10::kCUDA);
    const auto memcpy_kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
#else
    const auto memcpy_kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
#endif  // TORCH_HAS_CUDA
    CUDA_CHECK(
        cudaMemcpyAsync(outputs[i], tensor.data_ptr(), tensor.nbytes(), memcpy_kind, stream));
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t TorchModulePlugin::getSerializationSize() const {
  return serialized_size(data_type_) + serialized_size(node_ids_) + serialized_size(in_types_) +
         serialized_size(out_types_) + serialized_size(out_dims_);
}

void TorchModulePlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, node_ids_);
  serialize_value(&buffer, in_types_);
  serialize_value(&buffer, out_types_);
  serialize_value(&buffer, out_dims_);
}

bool TorchModulePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                  int nbInputs, int nbOutputs) {
  ASSERT(inOut && nbInputs > 0 && nbOutputs > 0 && pos < (nbInputs + nbOutputs));
  return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char* TorchModulePlugin::getPluginType() const { return TORCH_MODULE_PLUGIN_NAME; }

const char* TorchModulePlugin::getPluginVersion() const { return TORCH_MODULE_PLUGIN_VERSION; }

void TorchModulePlugin::destroy() { delete this; }

nvinfer1::IPluginV2DynamicExt* TorchModulePlugin::clone() const {
  return new TorchModulePlugin{data_type_, node_ids_, in_types_, out_types_, out_dims_};
}

void TorchModulePlugin::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* TorchModulePlugin::getPluginNamespace() const { return mPluginNamespace.c_str(); }

nvinfer1::DataType TorchModulePlugin::getOutputDataType(int index,
                                                        const nvinfer1::DataType* inputTypes,
                                                        int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0 && index >= 0 && index < out_types_.size());
  // output data type should be the same with input
  auto dtype = static_cast<nvinfer1::DataType>(out_types_[index]);
  if (data_type_ == nvinfer1::DataType::kHALF && dtype == nvinfer1::DataType::kFLOAT) {
    return nvinfer1::DataType::kHALF;
  }

  return dtype;
}

void TorchModulePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc* out,
                                        int nbOutputs) {
  ASSERT(data_type_ == nvinfer1::DataType::kFLOAT || data_type_ == nvinfer1::DataType::kHALF);
  // load original module to obtain nodes' meta data
  std::string path(TORCH_MODULE_PLUGIN_MODULE_PATH);
  fwd::torch_::TorchModule module;
#ifdef TORCH_HAS_CUDA
  ASSERT(module.Load(path, InferMode::FLOAT, c10::kCUDA));
#else
  ASSERT(module.Load(path, InferMode::FLOAT, c10::kCPU));
#endif  // TORCH_HAS_CUDA
  module.PostProcessGraph();

  // build id->node map of the original graph
  std::unordered_map<int, torch::jit::Node*> id_node_map;
  for (const auto& node : module.Graph()->nodes()) {
    id_node_map[node->outputs()[0]->unique()] = node;
  }

  // Create SubModule by cloning nodes with node_ids and id->node map
  sub_module_ = std::make_shared<fwd::torch_::TorchSubModule>();
  for (auto id : node_ids_) {
    ASSERT(sub_module_->AddNode(id_node_map[id]));
  }
  // Copy attributes which is needed by weight and bias of nodes
  sub_module_->AddAttributes(module.NamedAttributes());
  // Create module
  ASSERT(sub_module_->CreateModule());
  ASSERT(sub_module_->Inputs().size() == nbInputs);
  ASSERT(in_types_.size() == nbInputs);
  ASSERT(out_types_.size() == nbOutputs);

#ifdef TORCH_HAS_CUDA
  // if torch.is_cuda(), use cuda module
  sub_module_->ToCuda();
#else
  // allocate cpu input tensor
  p_inputs_.resize(nbInputs);
  for (int i = 0; i < nbInputs; ++i) {
    auto volume =
        std::accumulate(in[i].max.d, in[i].max.d + in[i].max.nbDims, 1, std::multiplies<int>());
    p_inputs_[i].Resize(volume);
  }
#endif  // TORCH_HAS_CUDA
}

int32_t TorchModulePlugin::initialize() { return 0; }

void TorchModulePlugin::terminate() {}

TorchModulePluginCreator::TorchModulePluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("module_path", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("node_ids", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* TorchModulePluginCreator::getPluginName() const { return TORCH_MODULE_PLUGIN_NAME; }

const char* TorchModulePluginCreator::getPluginVersion() const {
  return TORCH_MODULE_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* TorchModulePluginCreator::getFieldNames() { return &mFC; }

nvinfer1::IPluginV2DynamicExt* TorchModulePluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  int data_type{};
  std::vector<int> node_ids;
  std::vector<int> in_types;
  std::vector<int> out_types;
  std::vector<nvinfer1::Dims> out_dims;
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "node_ids")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      auto data = static_cast<const int*>(fields[i].data);
      node_ids.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "in_types")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      auto data = static_cast<const int*>(fields[i].data);
      in_types.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "out_types")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      auto data = static_cast<const int*>(fields[i].data);
      out_types.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "out_dims")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kDIMS);
      auto data = static_cast<const nvinfer1::Dims*>(fields[i].data);
      out_dims.assign(data, data + fields[i].length);
    } else {
      ASSERT(false);
    }
  }

  ASSERT(!out_dims.empty());

  auto obj = new TorchModulePlugin(static_cast<nvinfer1::DataType>(data_type), node_ids, in_types,
                                   out_types, out_dims);
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* TorchModulePluginCreator::deserializePlugin(const char* name,
                                                                           const void* serialData,
                                                                           size_t serialLength) {
  auto* obj = new TorchModulePlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
