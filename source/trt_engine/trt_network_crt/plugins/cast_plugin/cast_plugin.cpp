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

#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

#include <cuda_fp16.h>

#include <functional>
#include <numeric>
#include <string>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

FWD_TRT_NAMESPACE_BEGIN

CastPlugin::CastPlugin(nvinfer1::DataType data_type, nvinfer1::DataType output_type)
    : data_type_(data_type), output_type_(output_type) {}

CastPlugin::CastPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &output_type_);
}

CastPlugin::~CastPlugin() { terminate(); }

int CastPlugin::getNbOutputs() const { return 1; }

nvinfer1::DimsExprs CastPlugin::getOutputDimensions(int outputIndex,
                                                    const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                    nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1)
  return inputs[0];
}

size_t CastPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                    const nvinfer1::PluginTensorDesc* outputs,
                                    int nbOutputs) const {
  return 0;
}

int CastPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                        void* const* outputs, void* workspace, cudaStream_t stream) {
  // TODO(Ao Li): 目前暂时只支持向 float 的转换
  ASSERT(output_type_ == nvinfer1::DataType::kFLOAT || output_type_ == nvinfer1::DataType::kHALF);

  const auto volume = std::accumulate(inputDesc->dims.d, inputDesc->dims.d + inputDesc->dims.nbDims,
                                      1, std::multiplies<int64_t>());

  switch (data_type_) {
    case nvinfer1::DataType::kHALF:
      Cast<half, float>(static_cast<const half*>(inputs[0]), static_cast<float*>(outputs[0]),
                        volume);
      break;
    case nvinfer1::DataType::kINT8:
      Cast<int8_t, float>(static_cast<const int8_t*>(inputs[0]), static_cast<float*>(outputs[0]),
                          volume);
      break;
    case nvinfer1::DataType::kINT32:
      Cast<int, float>(static_cast<const int*>(inputs[0]), static_cast<float*>(outputs[0]), volume);
      break;
#if NV_TENSORRT_MAJOR >= 7
    case nvinfer1::DataType::kBOOL:
      Cast<bool, float>(static_cast<const bool*>(inputs[0]), static_cast<float*>(outputs[0]),
                        volume);
      break;
#endif // NV_TENSORRT_MAJOR >= 7
    default:
      break;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t CastPlugin::getSerializationSize() const {
  return serialized_size(data_type_) + serialized_size(output_type_);
}

void CastPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, output_type_);
}

bool CastPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                           int nbInputs, int nbOutputs) {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));
  return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[1].type == nvinfer1::DataType::kFLOAT;
}

const char* CastPlugin::getPluginType() const { return CAST_PLUGIN_NAME; }

const char* CastPlugin::getPluginVersion() const { return CAST_PLUGIN_VERSION; }

void CastPlugin::destroy() { delete this; }

nvinfer1::IPluginV2DynamicExt* CastPlugin::clone() const {
  return new CastPlugin{data_type_, output_type_};
}

void CastPlugin::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* CastPlugin::getPluginNamespace() const { return mPluginNamespace.c_str(); }

nvinfer1::DataType CastPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                 int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return output_type_;
}

void CastPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  for (int i = 0; i < nbInputs; i++) {
    for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
      // Do not support dynamic dimensions
      ASSERT(in[i].desc.dims.d[j] != -1);
    }
  }
}

int32_t CastPlugin::initialize() { return 0; }

void CastPlugin::terminate() {}

CastPluginCreator::CastPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("output_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* CastPluginCreator::getPluginName() const { return CAST_PLUGIN_NAME; }

const char* CastPluginCreator::getPluginVersion() const { return CAST_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* CastPluginCreator::getFieldNames() { return &mFC; }

nvinfer1::IPluginV2DynamicExt* CastPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  int data_type{}, output_type{};
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "output_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      output_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj = new CastPlugin(static_cast<nvinfer1::DataType>(data_type),
                            static_cast<nvinfer1::DataType>(output_type));
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* CastPluginCreator::deserializePlugin(const char* name,
                                                                    const void* serialData,
                                                                    size_t serialLength) {
  auto* obj = new CastPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
