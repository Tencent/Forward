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

#include "trt_engine/trt_network_crt/plugins/reduce_plugin/reduce_plugin.h"

#include <cuda_fp16.h>

#include <algorithm>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_REDUCE_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

// TODO(Ao Li): 暂时只实现了 reduce sum, 需要支持更多的 op

ReducePlugin::ReducePlugin(const std::vector<int>& reduce_dims, int keep_dim,
                           nvinfer1::DataType data_type, float power)
    : reduce_dims_(reduce_dims), keep_dim_(keep_dim), data_type_(data_type), power_(power) {}

ReducePlugin::ReducePlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &reduce_dims_);
  deserialize_value(&serialData, &serialLength, &keep_dim_);
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &power_);
}

ReducePlugin::~ReducePlugin() { terminate(); }

int ReducePlugin::getNbOutputs() const { return 1; }

nvinfer1::DimsExprs ReducePlugin::getOutputDimensions(int outputIndex,
                                                      const nvinfer1::DimsExprs* inputs,
                                                      int nbInputs,
                                                      nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1);

  nvinfer1::DimsExprs output;
  if (keep_dim_) {
#ifdef SHUFFLE_REDUCE
    output.nbDims = inputs[0].nbDims;
    for (int i = 0, k = 0; i < inputs[0].nbDims; ++i) {
      if (std::find(reduce_dims_.begin(), reduce_dims_.end(), i) != reduce_dims_.end()) {
        output.d[i] = exprBuilder.constant(1);
      } else {
        output.d[i] = inputs[0].d[k++];
      }
    }
#else
    output = inputs[0];
    for (auto dim : reduce_dims_) {
      output.d[dim] = exprBuilder.constant(1);
    }
#endif
  } else {
    output = inputs[0];
    output.nbDims -= reduce_dims_.size();
    for (int i = 0, k = 0; k < inputs[0].nbDims; ++k) {
      if (std::find(reduce_dims_.begin(), reduce_dims_.end(), k) == reduce_dims_.end()) {
        output.d[i++] = inputs[0].d[k];
      }
    }
  }

  return output;
}

int ReducePlugin::initialize() { return 0; }

void ReducePlugin::terminate() {}

size_t ReducePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                      const nvinfer1::PluginTensorDesc* outputs,
                                      int nbOutputs) const {
  return 0;
}

int ReducePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                          const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                          void* const* outputs, void* workspace, cudaStream_t stream) {
  const auto input_dim = inputDesc[0].dims;
  int64_t M = 1, N = 1, K = 1;

#ifdef SHUFFLE_REDUCE
  int i = 0;
  for (; i < input_dim.nbDims - reduce_dims_.size(); ++i) {
    M *= input_dim.d[i];
  }
  for (; i < input_dim.nbDims; ++i) {
    N *= input_dim.d[i];
  }
#else
  int i = 0;
  for (; i < reduce_dims_[0]; ++i) {
    M *= input_dim.d[i];
  }
  for (; i < reduce_dims_.back() + 1; ++i) {
    N *= input_dim.d[i];
  }
  for (; i < input_dim.nbDims; ++i) {
    K *= input_dim.d[i];
  }
#endif

  // std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

#ifdef ENABLE_REDUCE_FLOAT16
  switch (data_type_) {
#else
  switch (inputDesc[0].type) {
#endif
    case nvinfer1::DataType::kFLOAT:
      ReduceCuda<float>(static_cast<const float*>(inputs[0]), M, N, K,
                        static_cast<float*>(outputs[0]), power_, stream);
      break;

    case nvinfer1::DataType::kHALF:
      ReduceCuda<half>(static_cast<const half*>(inputs[0]), M, N, K, static_cast<half*>(outputs[0]),
                       power_, stream);
      break;
    default:
      getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "[Reduce] Unsupported input data type");
      return -1;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t ReducePlugin::getSerializationSize() const {
  return serialized_size(reduce_dims_) + serialized_size(keep_dim_) + serialized_size(data_type_) +
         serialized_size(power_);
}

void ReducePlugin::serialize(void* buffer) const {
  serialize_value(&buffer, reduce_dims_);
  serialize_value(&buffer, keep_dim_);
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, power_);
}

bool ReducePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                             int nbInputs, int nbOutputs) {
#ifdef ENABLE_REDUCE_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

const char* ReducePlugin::getPluginType() const { return REDUCE_PLUGIN_NAME; }

const char* ReducePlugin::getPluginVersion() const { return REDUCE_PLUGIN_VERSION; }

void ReducePlugin::destroy() { delete this; }

nvinfer1::IPluginV2DynamicExt* ReducePlugin::clone() const {
  return new ReducePlugin{reduce_dims_, keep_dim_, data_type_, power_};
}

void ReducePlugin::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* ReducePlugin::getPluginNamespace() const { return mPluginNamespace; }

nvinfer1::DataType ReducePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                   int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0 && index < 1);
  return inputTypes[0];
}

void ReducePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                   const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }
}

ReducePluginCreator::ReducePluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("reduce_dims", nullptr, nvinfer1::PluginFieldType::kINT32, 8));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("keep_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("power", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* ReducePluginCreator::getPluginName() const { return REDUCE_PLUGIN_NAME; }

const char* ReducePluginCreator::getPluginVersion() const { return REDUCE_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* ReducePluginCreator::getFieldNames() { return &mFC; }

nvinfer1::IPluginV2DynamicExt* ReducePluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  int keep_dim = 0;
  std::vector<int> reduce_dims;
  const nvinfer1::PluginField* fields = fc->fields;
  int data_type = 0;
  float power = 1.0f;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "reduce_dims")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      const auto* data = static_cast<const int*>(fields[i].data);
      reduce_dims.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "keep_dim")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      keep_dim = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "power")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      power = *(static_cast<const float*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj =
      new ReducePlugin(reduce_dims, keep_dim, static_cast<nvinfer1::DataType>(data_type), power);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

nvinfer1::IPluginV2DynamicExt* ReducePluginCreator::deserializePlugin(const char* name,
                                                                      const void* serialData,
                                                                      size_t serialLength) {
  auto* obj = new ReducePlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
