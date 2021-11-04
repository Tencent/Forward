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

#include "trt_engine/trt_network_crt/plugins/adaptive_lin_plugin/adaptive_lin_plugin.h"

#include <cuda_fp16.h>

#include <algorithm>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_ADAPTIVE_LIN_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

AdaptiveLINPlugin::AdaptiveLINPlugin(const nvinfer1::Dims &input_dim, float epsilon,
                                     nvinfer1::DataType data_type, int max_batch_size)
    : input_dim_(input_dim),
      epsilon_(epsilon),
      data_type_(data_type),
      initialized_(false),
      max_batch_size_(max_batch_size) {}

AdaptiveLINPlugin::AdaptiveLINPlugin(void const *serialData, size_t serialLength)
    : initialized_(false) {
  deserialize_value(&serialData, &serialLength, &input_dim_);
  deserialize_value(&serialData, &serialLength, &epsilon_);
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &max_batch_size_);
}

AdaptiveLINPlugin::~AdaptiveLINPlugin() { terminate(); }

int AdaptiveLINPlugin::getNbOutputs() const noexcept { return 2; }

nvinfer1::DimsExprs AdaptiveLINPlugin::getOutputDimensions(int outputIndex,
                                                           const nvinfer1::DimsExprs *inputs,
                                                           int nbInputs,
                                                           nvinfer1::IExprBuilder &exprBuilder) noexcept {
  const nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

bool AdaptiveLINPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut,
                                                  int nbInputs, int nbOutputs) noexcept {
#ifdef ENABLE_ADAPTIVE_LIN_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

void AdaptiveLINPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc *out,
                                        int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }

  if (initialized_) {
    return;
  }

  size_t n = max_batch_size_;
  const size_t c = input_dim_.d[1];

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in_mean_), n * c * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in_rstd_), n * c * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ln_mean_), n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ln_rstd_), n * sizeof(float)));

  initialized_ = true;
}

size_t AdaptiveLINPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                           const nvinfer1::PluginTensorDesc *outputs,
                                           int nbOutputs) const noexcept {
  return 0;
}

int AdaptiveLINPlugin::initialize() noexcept { return 0; }

void AdaptiveLINPlugin::terminate() noexcept {
  if (!initialized_) {
    return;
  }

  CUDA_CHECK(cudaFree(d_in_mean_));
  CUDA_CHECK(cudaFree(d_in_rstd_));
  CUDA_CHECK(cudaFree(d_ln_mean_));
  CUDA_CHECK(cudaFree(d_ln_rstd_));

  initialized_ = false;
}

int AdaptiveLINPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                               const nvinfer1::PluginTensorDesc *outputDesc,
                               const void *const *inputs, void *const *outputs, void *workspace,
                               cudaStream_t stream) noexcept {
  assert(initialized_);

  // input0 : input
  // output0 : layer norm
  // output1 : instance norm
  const auto input_dim = inputDesc[0].dims;
  ASSERT(input_dim.nbDims == 4);

  const int64_t N = input_dim.d[0];
  const int64_t C = input_dim.d[1];
  const int64_t H = input_dim.d[2];
  const int64_t W = input_dim.d[3];

  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_ln_mean_), 0, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_ln_rstd_), 0, N * sizeof(float)));

#ifdef ENABLE_ADAPTIVE_LIN_FLOAT16
  switch (data_type_) {
#else
  switch (inputDesc[0].type) {
#endif
    case nvinfer1::DataType::kFLOAT:
      AdaptiveLINCUDA<float>(static_cast<const float *>(inputs[0]), N, C, H * W, epsilon_,
                             d_in_mean_, d_in_rstd_, d_ln_mean_, d_ln_rstd_,
                             static_cast<float *>(outputs[1]), static_cast<float *>(outputs[0]),
                             stream);
      break;
    case nvinfer1::DataType::kHALF:
      AdaptiveLINCUDA<half>(static_cast<const half *>(inputs[0]), N, C, H * W, epsilon_, d_in_mean_,
                            d_in_rstd_, d_ln_mean_, d_ln_rstd_, static_cast<half *>(outputs[1]),
                            static_cast<half *>(outputs[0]), stream);
      break;
    default:
      getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                       "[Adaptive LIN] Unsupported input data type");
      return -1;
  }

  CUDA_CHECK(cudaGetLastError());
  return 0;
}

size_t AdaptiveLINPlugin::getSerializationSize() const noexcept {
  return serialized_size(input_dim_) + serialized_size(epsilon_) + serialized_size(data_type_) +
         serialized_size(max_batch_size_);
}

void AdaptiveLINPlugin::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, input_dim_);
  serialize_value(&buffer, epsilon_);
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, max_batch_size_);
}

const char *AdaptiveLINPlugin::getPluginType() const noexcept { return ADAPTIVE_LIN_PLUGIN_NAME; }

const char *AdaptiveLINPlugin::getPluginVersion() const noexcept { return ADAPTIVE_LIN_PLUGIN_VERSION; }

void AdaptiveLINPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt *AdaptiveLINPlugin::clone() const noexcept {
  return new AdaptiveLINPlugin(input_dim_, epsilon_, data_type_, max_batch_size_);
}

void AdaptiveLINPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char *AdaptiveLINPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType AdaptiveLINPlugin::getOutputDataType(int index,
                                                        const nvinfer1::DataType *inputTypes,
                                                        int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index < 2);
  return inputTypes[0];
}

AdaptiveLINPluginCreator::AdaptiveLINPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("input_dim", nullptr, nvinfer1::PluginFieldType::kDIMS, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("max_batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *AdaptiveLINPluginCreator::getPluginName() const noexcept { return ADAPTIVE_LIN_PLUGIN_NAME; }

const char *AdaptiveLINPluginCreator::getPluginVersion() const noexcept {
  return ADAPTIVE_LIN_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *AdaptiveLINPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2DynamicExt *AdaptiveLINPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  nvinfer1::Dims dims;
  float epsilon = 0.0f;
  int data_type = 0;
  int max_batch_size = -1;
  const nvinfer1::PluginField *fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char *attrName = fields[i].name;
    if (!strcmp(attrName, "input_dim")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kDIMS);
      dims = *(static_cast<const nvinfer1::Dims *>(fields[i].data));
    } else if (!strcmp(attrName, "epsilon")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      epsilon = *(static_cast<const float *>(fields[i].data));
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "max_batch_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      max_batch_size = *(static_cast<const int *>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj = new AdaptiveLINPlugin(dims, epsilon, static_cast<nvinfer1::DataType>(data_type),
                                   max_batch_size);
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt *AdaptiveLINPluginCreator::deserializePlugin(const char *name,
                                                                           const void *serialData,
                                                                           size_t serialLength) noexcept {
  auto *obj = new AdaptiveLINPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
