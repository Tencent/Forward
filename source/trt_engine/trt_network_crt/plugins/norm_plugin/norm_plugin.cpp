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

#include "trt_engine/trt_network_crt/plugins/norm_plugin/norm_plugin.h"

#include <cuda_fp16.h>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

FWD_TRT_NAMESPACE_BEGIN
// TODO(Paul Lu): 暂时只实现了 reduce sum, 需要支持更多的 op

NormPlugin::NormPlugin(const std::vector<int>& reduce_dims, int keep_dim,
                       nvinfer1::DataType data_type, float power)
    : reduce_dims_(reduce_dims),
      keep_dim_(keep_dim),
      data_type_(data_type),
      power_(power),
      initialized_(false) {}

NormPlugin::NormPlugin(void const* serialData, size_t serialLength) : initialized_(false) {
  deserialize_value(&serialData, &serialLength, &reduce_dims_);
  deserialize_value(&serialData, &serialLength, &keep_dim_);
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &power_);
}

NormPlugin::~NormPlugin() { terminate(); }

int NormPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs NormPlugin::getOutputDimensions(int outputIndex,
                                                    const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  ASSERT(nbInputs == 1);

  const nvinfer1::DimsExprs output = inputs[0];

  return output;
}

int NormPlugin::initialize() noexcept {
  if (initialized_) {
    return 0;
  }
  initialized_ = true;
  return 0;
}

void NormPlugin::terminate() noexcept {
  if (!initialized_) {
    return;
  }
  initialized_ = false;
}

size_t NormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                    const nvinfer1::PluginTensorDesc* outputs,
                                    int nbOutputs) const noexcept {
  return 0;
}

int NormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                        void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  const auto input_dim = inputDesc[0].dims;
  int64_t M = 1, N = 1, K = 1;

#ifdef SHUFFLE_NORM
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

  if (data_type_ == nvinfer1::DataType::kFLOAT) {
    NormCuda<float>(static_cast<const float*>(inputs[0]), M, N, K, static_cast<float*>(outputs[0]),
                    power_, stream);
  } else if (data_type_ == nvinfer1::DataType::kHALF) {
    NormCuda<__half>(static_cast<const __half*>(inputs[0]), M, N, K,
                     static_cast<__half*>(outputs[0]), power_, stream);
  } else {
    getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Unsupported input data type");
    return -1;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t NormPlugin::getSerializationSize() const noexcept {
  return serialized_size(reduce_dims_) + serialized_size(keep_dim_) + serialized_size(data_type_) +
         serialized_size(power_);
}

void NormPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, reduce_dims_);
  serialize_value(&buffer, keep_dim_);
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, power_);
}

bool NormPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                           int nbInputs, int nbOutputs) noexcept {
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

const char* NormPlugin::getPluginType() const noexcept { return NORM_PLUGIN_NAME; }

const char* NormPlugin::getPluginVersion() const noexcept { return NORM_PLUGIN_VERSION; }

void NormPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* NormPlugin::clone() const noexcept {
  return new NormPlugin{reduce_dims_, keep_dim_, data_type_, power_};
}

void NormPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* NormPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType NormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                 int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index < 1);
  return inputTypes[0];
}

void NormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }
}

NormPluginCreator::NormPluginCreator() {
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

const char* NormPluginCreator::getPluginName() const noexcept { return NORM_PLUGIN_NAME; }

const char* NormPluginCreator::getPluginVersion() const noexcept { return NORM_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* NormPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2DynamicExt* NormPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
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
      new NormPlugin(reduce_dims, keep_dim, static_cast<nvinfer1::DataType>(data_type), power);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

nvinfer1::IPluginV2DynamicExt* NormPluginCreator::deserializePlugin(const char* name,
                                                                    const void* serialData,
                                                                    size_t serialLength) noexcept {
  auto* obj = new NormPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
