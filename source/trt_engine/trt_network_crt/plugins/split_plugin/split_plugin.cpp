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

#include "trt_engine/trt_network_crt/plugins/split_plugin/split_plugin.h"

#include <cuda_fp16.h>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

FWD_TRT_NAMESPACE_BEGIN

SplitPlugin::SplitPlugin(int dim, const std::vector<int>& split_size, nvinfer1::DataType data_type)
    : dim_(dim), split_size_(split_size), data_type_(data_type), initialized_(false) {}

SplitPlugin::SplitPlugin(void const* serialData, size_t serialLength) : initialized_(false) {
  deserialize_value(&serialData, &serialLength, &dim_);
  deserialize_value(&serialData, &serialLength, &split_size_);
  deserialize_value(&serialData, &serialLength, &data_type_);
}

SplitPlugin::~SplitPlugin() { terminate(); }

int SplitPlugin::getNbOutputs() const noexcept { return split_size_.size(); }

nvinfer1::DimsExprs SplitPlugin::getOutputDimensions(int outputIndex,
                                                     const nvinfer1::DimsExprs* inputs,
                                                     int nbInputs,
                                                     nvinfer1::IExprBuilder& exprBuilder) noexcept {
  ASSERT(nbInputs == 1);
  nvinfer1::DimsExprs output(inputs[0]);
  output.d[dim_] = exprBuilder.constant(split_size_[outputIndex]);
  return output;
}

int SplitPlugin::initialize() noexcept {
  if (initialized_) {
    return 0;
  }

  // split size
  size_t bytes = sizeof(int) * split_size_.size();
  CUDA_CHECK(cudaMalloc(&d_split_size_, bytes));
  cudaMemcpy(d_split_size_, split_size_.data(), bytes, cudaMemcpyHostToDevice);

  std::vector<int> output_pos, output_off;
  for (size_t i = 0; i < split_size_.size(); ++i) {
    output_pos.insert(output_pos.end(), split_size_[i], i);
    for (int j = 0; j < split_size_[i]; ++j) {
      output_off.push_back(j);
    }
  }

  bytes = sizeof(int) * output_pos.size();
  CUDA_CHECK(cudaMalloc(&d_output_pos_, bytes));
  CUDA_CHECK(cudaMalloc(&d_output_off_, bytes));
  cudaMemcpy(d_output_pos_, output_pos.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_off_, output_off.data(), bytes, cudaMemcpyHostToDevice);

  // output pointers - pinned memory
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_outputs_),
                           sizeof(float*) * split_size_.size(), cudaHostAllocMapped));

  initialized_ = true;
  return 0;
}

void SplitPlugin::terminate() noexcept {
  if (!initialized_) {
    return;
  }

  CUDA_CHECK(cudaFree(d_split_size_));
  CUDA_CHECK(cudaFree(d_output_pos_));
  CUDA_CHECK(cudaFree(d_output_off_));
  CUDA_CHECK(cudaFreeHost(pinned_outputs_));

  initialized_ = false;
}

size_t SplitPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                     const nvinfer1::PluginTensorDesc* outputs,
                                     int nbOutputs) const noexcept {
  return 0;
}

int SplitPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                         const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                         void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  for (size_t i = 0; i < split_size_.size(); ++i) {
    pinned_outputs_[i] = static_cast<float*>(outputs[i]);
  }

  if (data_type_ == nvinfer1::DataType::kFLOAT) {
    const TensorInfo<float> input_tensor(static_cast<const float*>(inputs[0]), inputDesc[0].dims);
    SplitCuda<float>(input_tensor, pinned_outputs_, d_split_size_, d_output_pos_, d_output_off_,
                     dim_, stream);
  } else if (data_type_ == nvinfer1::DataType::kHALF) {
    const TensorInfo<__half> input_tensor(static_cast<const __half*>(inputs[0]), inputDesc[0].dims);
    SplitCuda<__half>(input_tensor, reinterpret_cast<__half* const*>(pinned_outputs_),
                      d_split_size_, d_output_pos_, d_output_off_, dim_, stream);
  } else {
    getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Unsupported input data type");
    return -1;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t SplitPlugin::getSerializationSize() const noexcept {
  return serialized_size(dim_) + serialized_size(split_size_) + serialized_size(data_type_);
}

void SplitPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, dim_);
  serialize_value(&buffer, split_size_);
  serialize_value(&buffer, data_type_);
}

bool SplitPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                            int nbInputs, int nbOutputs) noexcept {
  //        return (inOut[pos].type == nvinfer1::DataType::kFLOAT
  //            && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

const char* SplitPlugin::getPluginType() const noexcept { return SPLIT_PLUGIN_NAME; }

const char* SplitPlugin::getPluginVersion() const noexcept { return SPLIT_PLUGIN_VERSION; }

void SplitPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* SplitPlugin::clone() const noexcept {
  return new SplitPlugin{dim_, split_size_, data_type_};
}

void SplitPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* SplitPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType SplitPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                  int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index < split_size_.size());
  return inputTypes[0];
}

void SplitPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  for (int i = 0; i < nbInputs; i++) {
    for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
      // Do not support dynamic dimensions
      ASSERT(in[i].desc.dims.d[j] != -1);
    }
  }
}

SplitPluginCreator::SplitPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("split_size", nullptr, nvinfer1::PluginFieldType::kINT32, 8));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* SplitPluginCreator::getPluginName() const noexcept { return SPLIT_PLUGIN_NAME; }

const char* SplitPluginCreator::getPluginVersion() const noexcept { return SPLIT_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* SplitPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2DynamicExt* SplitPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  int dim = -1;
  std::vector<int> split_size;
  const nvinfer1::PluginField* fields = fc->fields;
  int data_type = 0;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "dim")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      dim = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "split_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      const auto* data = static_cast<const int*>(fields[i].data);
      split_size.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj = new SplitPlugin(dim, split_size, static_cast<nvinfer1::DataType>(data_type));
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

nvinfer1::IPluginV2DynamicExt* SplitPluginCreator::deserializePlugin(const char* name,
                                                                     const void* serialData,
                                                                     size_t serialLength) noexcept {
  auto* obj = new SplitPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
