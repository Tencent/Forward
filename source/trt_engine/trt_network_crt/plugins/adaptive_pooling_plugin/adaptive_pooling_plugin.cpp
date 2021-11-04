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

#include "trt_engine/trt_network_crt/plugins/adaptive_pooling_plugin/adaptive_pooling_plugin.h"

#include <cuda_fp16.h>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_ADAPTIVE_POOLING_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

AdaptivePoolingPlugin::AdaptivePoolingPlugin(const std::vector<int> &output_size, int type,
                                             nvinfer1::DataType data_type)
    : output_size_(output_size), type_(type), data_type_(data_type) {}

AdaptivePoolingPlugin::AdaptivePoolingPlugin(void const *serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &output_size_);
  deserialize_value(&serialData, &serialLength, &type_);
  deserialize_value(&serialData, &serialLength, &data_type_);
}

AdaptivePoolingPlugin::~AdaptivePoolingPlugin() { terminate(); }

int AdaptivePoolingPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs AdaptivePoolingPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  // TODO(Ao Li): Add 1D supported
  ASSERT(inputs[0].nbDims == 4 && output_size_.size() == 2 ||
         inputs[0].nbDims == 5 && output_size_.size() == 3);

  nvinfer1::DimsExprs output(inputs[0]);

  output.d[2] = exprBuilder.constant(output_size_[0]);
  output.d[3] = exprBuilder.constant(output_size_[1]);

  if (output_size_.size() == 3) {
    output.d[4] = exprBuilder.constant(output_size_[2]);
  }

  return output;
}

bool AdaptivePoolingPlugin::supportsFormatCombination(int pos,
                                                      const nvinfer1::PluginTensorDesc *inOut,
                                                      int nbInputs, int nbOutputs) noexcept {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));
#ifdef ENABLE_ADAPTIVE_POOLING_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

void AdaptivePoolingPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                                            int nbInputs,
                                            const nvinfer1::DynamicPluginTensorDesc *out,
                                            int nbOutputs) noexcept {}

size_t AdaptivePoolingPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                               int nbInputs,
                                               const nvinfer1::PluginTensorDesc *outputs,
                                               int nbOutputs) const noexcept {
  return 0;
}

int AdaptivePoolingPlugin::initialize() noexcept { return 0; }

void AdaptivePoolingPlugin::terminate() noexcept {}

int AdaptivePoolingPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs, void *const *outputs, void *workspace,
                                   cudaStream_t stream) noexcept {
  if (output_size_.size() == 2) {  // 2d
#ifdef ENABLE_ADAPTIVE_POOLING_FLOAT16
    switch (data_type_) {
#else
    switch (inputDesc[0].type) {
#endif
      case nvinfer1::DataType::kFLOAT: {
        TensorInfo<float> output_tensor{static_cast<float *>(outputs[0]), outputDesc[0].dims};
        AdaptivePooling2DCuda<float>({static_cast<const float *>(inputs[0]), inputDesc[0].dims},
                                     output_tensor, output_size_, PoolingOperation(type_), stream);
        break;
      }
      case nvinfer1::DataType::kHALF: {
        TensorInfo<half> output_tensor{static_cast<half *>(outputs[0]), outputDesc[0].dims};
        AdaptivePooling2DCuda<half>({static_cast<const half *>(inputs[0]), inputDesc[0].dims},
                                    output_tensor, output_size_, PoolingOperation(type_), stream);
        break;
      }
      default: {
        getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                         "[Adaptive Pooling] Unsupported input data type");
        return -1;
      }
    }
  } else if (output_size_.size() == 3) {
#ifdef ENABLE_ADAPTIVE_POOLING_FLOAT16
    switch (data_type_) {
#else
    switch (inputDesc[0].type) {
#endif
      case nvinfer1::DataType::kFLOAT: {
        TensorInfo<float> output_tensor{static_cast<float *>(outputs[0]), outputDesc[0].dims};
        AdaptivePooling3DCuda<float>({static_cast<const float *>(inputs[0]), inputDesc[0].dims},
                                     output_tensor, output_size_, PoolingOperation(type_), stream);
        break;
      }
      case nvinfer1::DataType::kHALF: {
        TensorInfo<half> output_tensor{static_cast<half *>(outputs[0]), outputDesc[0].dims};
        AdaptivePooling3DCuda<half>({static_cast<const half *>(inputs[0]), inputDesc[0].dims},
                                    output_tensor, output_size_, PoolingOperation(type_), stream);
        break;
      }
      default: {
        getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                         "[Adaptive Pooling]Unsupported input data type");
        return -1;
      }
    }
  } else {
    getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                     "AdaptivePooling do not support output size > 3");
    return -1;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}  // namespace trt

size_t AdaptivePoolingPlugin::getSerializationSize() const noexcept {
  return serialized_size(output_size_) + serialized_size(type_) + serialized_size(data_type_);
}

void AdaptivePoolingPlugin::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, output_size_);
  serialize_value(&buffer, type_);
  serialize_value(&buffer, data_type_);
}

const char *AdaptivePoolingPlugin::getPluginType() const noexcept { return ADAPTIVE_POOLING_PLUGIN_NAME; }

const char *AdaptivePoolingPlugin::getPluginVersion() const noexcept {
  return ADAPTIVE_POOLING_PLUGIN_VERSION;
}

void AdaptivePoolingPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt *AdaptivePoolingPlugin::clone() const noexcept {
  return new AdaptivePoolingPlugin(output_size_, type_, data_type_);
}

void AdaptivePoolingPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char *AdaptivePoolingPlugin::getPluginNamespace() const noexcept { return mPluginNamespace.c_str(); }

nvinfer1::DataType AdaptivePoolingPlugin::getOutputDataType(int index,
                                                            const nvinfer1::DataType *inputTypes,
                                                            int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

AdaptivePoolingPluginCreator::AdaptivePoolingPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("output_size", nullptr, nvinfer1::PluginFieldType::kINT32, 3));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *AdaptivePoolingPluginCreator::getPluginName() const noexcept {
  return ADAPTIVE_POOLING_PLUGIN_NAME;
}

const char *AdaptivePoolingPluginCreator::getPluginVersion() const noexcept {
  return ADAPTIVE_POOLING_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *AdaptivePoolingPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt *AdaptivePoolingPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  std::vector<int> output_size{};
  int type = 0;
  int data_type = 0;
  const nvinfer1::PluginField *fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char *attrName = fields[i].name;
    if (!strcmp(attrName, "output_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      const auto *data = static_cast<const int *>(fields[i].data);
      output_size.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      type = *(static_cast<const int *>(fields[i].data));
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int *>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj =
      new AdaptivePoolingPlugin(output_size, type, static_cast<nvinfer1::DataType>(data_type));
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt *AdaptivePoolingPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  auto *obj = new AdaptivePoolingPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
