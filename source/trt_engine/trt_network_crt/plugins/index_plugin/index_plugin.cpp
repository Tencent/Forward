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

#include "trt_engine/trt_network_crt/plugins/index_plugin/index_plugin.h"

#include <algorithm>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"
#include "trt_engine/trt_network_crt/plugins/common/trt_tensor_info.h"

FWD_TRT_NAMESPACE_BEGIN

bool IndexPlugin::CheckIndexPosition(int* index_dims) {
  bool init = false;
  bool gap = false;
  for (int i = 0; i < nb_dims_; i++) {
    if (index_dims[i]) {
      if (init && gap) {
        return true;
      } else {
        init = true;
      }
    } else if (init) {
      gap = true;
    }
  }
  return false;
}

IndexPlugin::IndexPlugin(const int* index_data, const int* index_position, int nb_dims,
                         int nb_index_dims, int nb_index)
    : nb_dims_(nb_dims), nb_index_dims_(nb_index_dims), nb_index_(nb_index), initialized_(false) {
  h_pos_.resize(nb_dims_);
  h_data_.resize(nb_index_dims_ * nb_index_);
  memcpy(h_pos_.data(), index_position, nb_dims_ * sizeof(int));
  memcpy(h_data_.data(), index_data, nb_index_dims_ * nb_index_ * sizeof(int));

  put_in_the_front_ = CheckIndexPosition(h_pos_.data());
}

IndexPlugin::IndexPlugin(void const* serialData, size_t serialLength) : initialized_(false) {
  deserialize_value(&serialData, &serialLength, &nb_dims_);
  deserialize_value(&serialData, &serialLength, &nb_index_dims_);
  deserialize_value(&serialData, &serialLength, &nb_index_);
  // h_data_.resize(nb_index_dims_ * nb_index_);
  deserialize_value(&serialData, &serialLength, &h_data_);
  // h_pos_.resize(nb_dims_);
  deserialize_value(&serialData, &serialLength, &h_pos_);

  put_in_the_front_ = CheckIndexPosition(h_pos_.data());
}

IndexPlugin::~IndexPlugin() { terminate(); }

int IndexPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs IndexPlugin::getOutputDimensions(int outputIndex,
                                                     const nvinfer1::DimsExprs* inputs,
                                                     int nbInputs,
                                                     nvinfer1::IExprBuilder& exprBuilder) noexcept {
  ASSERT(inputs[0].nbDims == nb_dims_);

  nvinfer1::DimsExprs output;

  output.nbDims = nb_dims_ - nb_index_dims_ + 1;

  if (put_in_the_front_) {
    int output_pos = 0;
    output.d[output_pos] = exprBuilder.constant(nb_index_);
    for (int i = 0; i < nb_dims_; i++) {
      if (!h_pos_[i]) {
        output.d[++output_pos] = inputs[0].d[i];
      }
    }
  } else {
    int input_pos = 0;
    for (int i = 0; i < output.nbDims; i++) {
      if (h_pos_[input_pos]) {
        output.d[i] = exprBuilder.constant(nb_index_);
        while (input_pos < inputs[0].nbDims && h_pos_[input_pos]) ++input_pos;
      } else {
        output.d[i] = inputs[0].d[input_pos];
        ++input_pos;
      }
    }
  }

  return output;
}

bool IndexPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                            int nbInputs, int nbOutputs) noexcept {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));

  return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

void IndexPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }

  if (initialized_) {
    return;
  }

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_data_), nb_index_dims_ * nb_index_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pos_), nb_dims_ * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_data_, h_data_.data(), nb_index_dims_ * nb_index_ * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos_, h_pos_.data(), nb_dims_ * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input_dims_), nb_dims_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_dims_), nb_dims_ * sizeof(int)));
  initialized_ = true;
}

size_t IndexPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                     const nvinfer1::PluginTensorDesc* outputs,
                                     int nbOutputs) const noexcept {
  return 0;
}

int IndexPlugin::initialize() noexcept { return 0; }

void IndexPlugin::terminate() noexcept {
  if (!initialized_) {
    return;
  }
  CUDA_CHECK(cudaFree(d_data_));
  CUDA_CHECK(cudaFree(d_pos_));
  CUDA_CHECK(cudaFree(d_input_dims_));
  CUDA_CHECK(cudaFree(d_output_dims_));
  initialized_ = false;
}

int IndexPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                         const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                         void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
    // TODO(Paul Lu): 在enqueue的时候做是担心维度变化，目前还不支持dynamic
    // shape
    CUDA_CHECK(cudaMemcpy(d_input_dims_, inputDesc[0].dims.d, nb_dims_ * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_dims_, outputDesc[0].dims.d, nb_dims_ * sizeof(int),
                          cudaMemcpyHostToDevice));

    IndexCuda(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), d_data_,
              d_input_dims_, d_output_dims_, d_pos_, inputDesc[0].dims.nbDims,
              outputDesc[0].dims.nbDims, nb_index_, put_in_the_front_, stream);
  } else {
    getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Unsupported input data type");
    return -1;
  }
  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t IndexPlugin::getSerializationSize() const noexcept {
  return serialized_size(nb_index_) + serialized_size(nb_index_dims_) + serialized_size(nb_dims_) +
         serialized_size(h_pos_) + serialized_size(h_data_);
}

void IndexPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, nb_dims_);
  serialize_value(&buffer, nb_index_dims_);
  serialize_value(&buffer, nb_index_);
  serialize_value(&buffer, h_data_);
  serialize_value(&buffer, h_pos_);
}

const char* IndexPlugin::getPluginType() const noexcept { return INDEX_PLUGIN_NAME; }

const char* IndexPlugin::getPluginVersion() const noexcept { return INDEX_PLUGIN_VERSION; }

void IndexPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* IndexPlugin::clone() const noexcept {
  return new IndexPlugin(h_data_.data(), h_pos_.data(), nb_dims_, nb_index_dims_, nb_index_);
}

void IndexPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* IndexPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType IndexPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                  int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

IndexPluginCreator::IndexPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("nbDims", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("nbIndexDims", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("nbIndex", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("pos", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* IndexPluginCreator::getPluginName() const noexcept { return INDEX_PLUGIN_NAME; }

const char* IndexPluginCreator::getPluginVersion() const noexcept { return INDEX_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* IndexPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2DynamicExt* IndexPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  int nb_dims;
  int nb_index_dims;
  int nb_index;
  const int* data;
  const int* pos;
  int type = 0;
  const nvinfer1::PluginField* fields = fc->fields;
  ASSERT(fc->nbFields == 5);
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "nbDims")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      nb_dims = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "nbIndexDims")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      nb_index_dims = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "nbIndex")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      nb_index = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "data")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data = static_cast<const int*>(fields[i].data);
    } else if (!strcmp(attrName, "pos")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      pos = static_cast<const int*>(fields[i].data);
    } else {
      ASSERT(false);
    }
  }

  auto obj = new IndexPlugin(data, pos, nb_dims, nb_index_dims, nb_index);
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* IndexPluginCreator::deserializePlugin(const char* name,
                                                                     const void* serialData,
                                                                     size_t serialLength) noexcept {
  auto* obj = new IndexPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
