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

#include "trt_engine/trt_network_crt/plugins/reflection_padding_plugin/reflection_padding_plugin.h"

#include <cuda_fp16.h>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_REFLECTION_PADDING_2D_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

ReflectionPadding2DPlugin::ReflectionPadding2DPlugin(const std::vector<int>& padding_size,
                                                     nvinfer1::DataType data_type)
    : padding_size_(padding_size), data_type_(data_type) {
  ASSERT(padding_size.size() == 4);
}

ReflectionPadding2DPlugin::ReflectionPadding2DPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &padding_size_);
  deserialize_value(&serialData, &serialLength, &data_type_);
}

ReflectionPadding2DPlugin::~ReflectionPadding2DPlugin() { terminate(); }

// ReflectionPadding2DPlugin returns one output.
int ReflectionPadding2DPlugin::getNbOutputs() const { return 1; }

nvinfer1::DimsExprs ReflectionPadding2DPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output(inputs[0]);
  output.d[2] = exprBuilder.constant(inputs[0].d[2]->getConstantValue() + padding_size_[2] +
                                     padding_size_[3]);
  output.d[3] = exprBuilder.constant(inputs[0].d[3]->getConstantValue() + padding_size_[0] +
                                     padding_size_[1]);

  return output;
}

int ReflectionPadding2DPlugin::initialize() { return 0; }

void ReflectionPadding2DPlugin::terminate() {}

size_t ReflectionPadding2DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                   int nbInputs,
                                                   const nvinfer1::PluginTensorDesc* outputs,
                                                   int nbOutputs) const {
  return 0;
}

int ReflectionPadding2DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                       const nvinfer1::PluginTensorDesc* outputDesc,
                                       const void* const* inputs, void* const* outputs,
                                       void* workspace, cudaStream_t stream) {
  const nvinfer1::Dims input_dims = inputDesc[0].dims;

#ifdef ENABLE_REFLECTION_PADDING_2D_FLOAT16
  switch (data_type_) {
#else
  switch (inputDesc[0].type) {
#endif
    case nvinfer1::DataType::kFLOAT:
      ReflectionPad2D<float>(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]),
                             input_dims, padding_size_, stream);
      break;
    case nvinfer1::DataType::kHALF:
      ReflectionPad2D<half>(static_cast<const half*>(inputs[0]), static_cast<half*>(outputs[0]),
                            input_dims, padding_size_, stream);
      break;
    default:
      getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                       "[Reflection Padding] Unsupported input data type");
      return -1;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t ReflectionPadding2DPlugin::getSerializationSize() const {
  return (serialized_size(padding_size_) + serialized_size(data_type_));
}

void ReflectionPadding2DPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, padding_size_);
  serialize_value(&buffer, data_type_);
}

bool ReflectionPadding2DPlugin::supportsFormatCombination(int pos,
                                                          const nvinfer1::PluginTensorDesc* inOut,
                                                          int nbInputs, int nbOutputs) {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));

#ifdef ENABLE_REFLECTION_PADDING_2D_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

const char* ReflectionPadding2DPlugin::getPluginType() const {
  return REFLECTION_PADDING_2D_PLUGIN_NAME;
}

const char* ReflectionPadding2DPlugin::getPluginVersion() const {
  return REFLECTION_PADDING_2D_PLUGIN_VERSION;
}

void ReflectionPadding2DPlugin::destroy() { delete this; }

nvinfer1::IPluginV2DynamicExt* ReflectionPadding2DPlugin::clone() const {
  return new ReflectionPadding2DPlugin(padding_size_, data_type_);
}

// Set plugin namespace
void ReflectionPadding2DPlugin::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* ReflectionPadding2DPlugin::getPluginNamespace() const {
  return mPluginNamespace.c_str();
}

nvinfer1::DataType ReflectionPadding2DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void ReflectionPadding2DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                                int nbInputs,
                                                const nvinfer1::DynamicPluginTensorDesc* out,
                                                int nbOutputs) {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }
}

// ReflectionPadding2DPluginCreator methods
ReflectionPadding2DPluginCreator::ReflectionPadding2DPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("padding_size", nullptr, nvinfer1::PluginFieldType::kINT32, 4));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* ReflectionPadding2DPluginCreator::getPluginName() const {
  return REFLECTION_PADDING_2D_PLUGIN_NAME;
}

const char* ReflectionPadding2DPluginCreator::getPluginVersion() const {
  return REFLECTION_PADDING_2D_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* ReflectionPadding2DPluginCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt* ReflectionPadding2DPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  std::vector<int> padding_size;
  int data_type = 0;
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;

    if (!strcmp(attrName, "padding_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      const auto* data = static_cast<const int*>(fields[i].data);
      padding_size.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  ReflectionPadding2DPlugin* obj =
      new ReflectionPadding2DPlugin(padding_size, static_cast<nvinfer1::DataType>(data_type));
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

nvinfer1::IPluginV2DynamicExt* ReflectionPadding2DPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  ReflectionPadding2DPlugin* obj = new ReflectionPadding2DPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
