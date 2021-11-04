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

#include "trt_engine/trt_network_crt/plugins/constant_pad_plugin/constant_pad_plugin.h"

#include <cuda_fp16.h>

#include <string>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"
#include "trt_engine/trt_network_crt/plugins/common/trt_tensor_info.h"

// #define ENABLE_CONSTANT_PAD_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

ConstantPadPlugin::ConstantPadPlugin(const std::vector<int>& padding_dims, float constant,
                                     nvinfer1::DataType data_type)
    : padding_dims_(padding_dims), constant_(constant), data_type_(data_type) {}

ConstantPadPlugin::ConstantPadPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &padding_dims_);
  deserialize_value(&serialData, &serialLength, &constant_);
  deserialize_value(&serialData, &serialLength, &data_type_);
}

ConstantPadPlugin::~ConstantPadPlugin() { terminate(); }

int ConstantPadPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs ConstantPadPlugin::getOutputDimensions(int outputIndex,
                                                           const nvinfer1::DimsExprs* inputs,
                                                           int nbInputs,
                                                           nvinfer1::IExprBuilder& exprBuilder) noexcept {
  ASSERT(inputs[0].nbDims == 4 && padding_dims_.size() == 4 ||
         inputs[0].nbDims == 5 && padding_dims_.size() == 6);

  nvinfer1::DimsExprs output(inputs[0]);

  if (padding_dims_.size() == 4) {
    output.d[2] = exprBuilder.constant(inputs[0].d[2]->getConstantValue() + padding_dims_[2] +
                                       padding_dims_[3]);
    output.d[3] = exprBuilder.constant(inputs[0].d[3]->getConstantValue() + padding_dims_[0] +
                                       padding_dims_[1]);
  } else if (padding_dims_.size() == 6) {
    output.d[2] = exprBuilder.constant(inputs[0].d[2]->getConstantValue() + padding_dims_[4] +
                                       padding_dims_[5]);
    output.d[3] = exprBuilder.constant(inputs[0].d[3]->getConstantValue() + padding_dims_[2] +
                                       padding_dims_[3]);
    output.d[4] = exprBuilder.constant(inputs[0].d[4]->getConstantValue() + padding_dims_[0] +
                                       padding_dims_[1]);
  }

  return output;
}

int ConstantPadPlugin::initialize() noexcept { return 0; }

void ConstantPadPlugin::terminate() noexcept {}

size_t ConstantPadPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                           const nvinfer1::PluginTensorDesc* outputs,
                                           int nbOutputs) const noexcept {
  return 0;
}

int ConstantPadPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                               const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) noexcept {
  nvinfer1::Dims input_dims = inputDesc[0].dims;

  if (input_dims.nbDims == 4) {
#ifdef ENABLE_CONSTANT_PAD_FLOAT16
    switch (data_type_) {
#else
    switch (inputDesc[0].type) {
#endif
      case nvinfer1::DataType::kFLOAT:
        ConstantPad2D<float>(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]),
                             constant_, input_dims, padding_dims_, stream);
        break;
      case nvinfer1::DataType::kHALF:
        ConstantPad2D<half>(static_cast<const half*>(inputs[0]), static_cast<half*>(outputs[0]),
                            constant_, input_dims, padding_dims_, stream);
        break;
      default:
        getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                         "[Constant Pad] Unsupported input data type");
        return -1;
    }
  } else if (input_dims.nbDims == 5) {
#ifdef ENABLE_CONSTANT_PAD_FLOAT16
    switch (data_type_) {
#else
    switch (inputDesc[0].type) {
#endif
      case nvinfer1::DataType::kFLOAT:
        ConstantPad3D<float>(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]),
                             constant_, input_dims, padding_dims_, stream);
        break;
      case nvinfer1::DataType::kHALF:
        ConstantPad3D<half>(static_cast<const half*>(inputs[0]), static_cast<half*>(outputs[0]),
                            constant_, input_dims, padding_dims_, stream);
        break;
      default:
        getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                         "[Constant Pad] Unsupported input data type");
        return -1;
    }
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t ConstantPadPlugin::getSerializationSize() const noexcept {
  return serialized_size(padding_dims_) + serialized_size(constant_) + serialized_size(data_type_);
}

void ConstantPadPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, padding_dims_);
  serialize_value(&buffer, constant_);
  serialize_value(&buffer, data_type_);
}

bool ConstantPadPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                  int nbInputs, int nbOutputs) noexcept {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));
#ifdef ENABLE_CONSTANT_PAD_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

const char* ConstantPadPlugin::getPluginType() const noexcept { return CONSTANT_PAD_PLUGIN_NAME; }

const char* ConstantPadPlugin::getPluginVersion() const noexcept { return CONSTANT_PAD_PLUGIN_VERSION; }

void ConstantPadPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* ConstantPadPlugin::clone() const noexcept {
  return new ConstantPadPlugin{padding_dims_, constant_, data_type_};
}

void ConstantPadPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* ConstantPadPlugin::getPluginNamespace() const noexcept { return mPluginNamespace.c_str(); }

nvinfer1::DataType ConstantPadPlugin::getOutputDataType(int index,
                                                        const nvinfer1::DataType* inputTypes,
                                                        int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void ConstantPadPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc* out,
                                        int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }
}

ConstantPadPluginCreator::ConstantPadPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("padding_dims", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("constant", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* ConstantPadPluginCreator::getPluginName() const noexcept { return CONSTANT_PAD_PLUGIN_NAME; }

const char* ConstantPadPluginCreator::getPluginVersion() const noexcept {
  return CONSTANT_PAD_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* ConstantPadPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2DynamicExt* ConstantPadPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::vector<int> padding_dims{};
  float constant{};
  int data_type{};
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "padding_dims")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      const auto* data = static_cast<const int*>(fields[i].data);
      padding_dims.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "constant")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      constant = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj =
      new ConstantPadPlugin(padding_dims, constant, static_cast<nvinfer1::DataType>(data_type));
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* ConstantPadPluginCreator::deserializePlugin(const char* name,
                                                                           const void* serialData,
                                                                           size_t serialLength) noexcept {
  auto* obj = new ConstantPadPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
