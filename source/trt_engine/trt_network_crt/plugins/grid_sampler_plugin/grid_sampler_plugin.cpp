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

#include "trt_engine/trt_network_crt/plugins/grid_sampler_plugin/grid_sampler_plugin.h"

#include <cuda_fp16.h>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_GRID_SAMPLER_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

GridSamplerPlugin::GridSamplerPlugin(int interpolation_mode, int padding_mode, int align_corners,
                                     nvinfer1::DataType data_type)
    : interpolation_mode_(interpolation_mode),
      padding_mode_(padding_mode),
      align_corners_(align_corners),
      data_type_(data_type) {}

GridSamplerPlugin::GridSamplerPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &interpolation_mode_);
  deserialize_value(&serialData, &serialLength, &padding_mode_);
  deserialize_value(&serialData, &serialLength, &align_corners_);
  deserialize_value(&serialData, &serialLength, &data_type_);
}

GridSamplerPlugin::~GridSamplerPlugin() { terminate(); }

int GridSamplerPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs GridSamplerPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // [N, C, H, W], [N, H', W', 2] -> [N, C, H', W']
  ASSERT(nbInputs == 2);
  assert(inputs[0].nbDims == 4 || inputs[0].nbDims == 5);
  assert(inputs[1].nbDims == 4 || inputs[1].nbDims == 5);

  nvinfer1::DimsExprs output(inputs[0]);
  output.d[2] = inputs[1].d[1];
  output.d[3] = inputs[1].d[2];

  if (inputs[0].nbDims == 5) {
    // [N, C, D, H, W], [N, D', H', W', 2] -> [N, C, D', H', W']
    output.d[4] = inputs[1].d[3];
  }

  return output;
}

int GridSamplerPlugin::initialize() noexcept { return 0; }

void GridSamplerPlugin::terminate() noexcept {}

size_t GridSamplerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                           const nvinfer1::PluginTensorDesc* outputs,
                                           int nbOutputs) const noexcept {
  return 0;
}

int GridSamplerPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                               const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) noexcept {
#ifdef ENABLE_GRID_SAMPLER_FLOAT16
  nvinfer1::DataType type = data_type_;
#else
  nvinfer1::DataType type = inputDesc[0].type;
#endif
  if (inputDesc[0].dims.nbDims == 4) {
    if (type == nvinfer1::DataType::kFLOAT) {
      TensorInfo<float> output_tensor{static_cast<float*>(outputs[0]), outputDesc[0].dims};
      GridSampler2DCuda<float>({static_cast<const float*>(inputs[0]), inputDesc[0].dims},
                               {static_cast<const float*>(inputs[1]), inputDesc[1].dims},
                               output_tensor, interpolation_mode_, padding_mode_, align_corners_,
                               stream);
    } else if (type == nvinfer1::DataType::kHALF) {
      TensorInfo<__half> output_tensor{static_cast<__half*>(outputs[0]), outputDesc[0].dims};
      GridSampler2DCuda<__half>({static_cast<const __half*>(inputs[0]), inputDesc[0].dims},
                                {static_cast<const __half*>(inputs[1]), inputDesc[1].dims},
                                output_tensor, interpolation_mode_, padding_mode_, align_corners_,
                                stream);
    } else {
      getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                       "[Grid Sampler] Unsupported input data type");
      return -1;
    }
  } else if (inputDesc[0].dims.nbDims == 5) {
    if (type == nvinfer1::DataType::kFLOAT) {
      TensorInfo<float> output_tensor{static_cast<float*>(outputs[0]), outputDesc[0].dims};
      GridSampler3DCuda<float>({static_cast<const float*>(inputs[0]), inputDesc[0].dims},
                               {static_cast<const float*>(inputs[1]), inputDesc[1].dims},
                               output_tensor, interpolation_mode_, padding_mode_, align_corners_,
                               stream);
    } else if (type == nvinfer1::DataType::kHALF) {
      TensorInfo<__half> output_tensor{static_cast<__half*>(outputs[0]), outputDesc[0].dims};
      GridSampler3DCuda<__half>({static_cast<const __half*>(inputs[0]), inputDesc[0].dims},
                                {static_cast<const __half*>(inputs[1]), inputDesc[1].dims},
                                output_tensor, interpolation_mode_, padding_mode_, align_corners_,
                                stream);
    } else {
      getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                       "[Grid Sampler] Unsupported input data type");
      return -1;
    }
  } else {
    getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                     "GridSampler do not support input dims > 5");
    return -1;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t GridSamplerPlugin::getSerializationSize() const noexcept {
  return serialized_size(interpolation_mode_) + serialized_size(padding_mode_) +
         serialized_size(align_corners_) + serialized_size(data_type_);
}

void GridSamplerPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, interpolation_mode_);
  serialize_value(&buffer, padding_mode_);
  serialize_value(&buffer, align_corners_);
  serialize_value(&buffer, data_type_);
}

bool GridSamplerPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                  int nbInputs, int nbOutputs) noexcept {
  ASSERT(inOut && nbInputs == 2 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));

#ifdef ENABLE_GRID_SAMPLER_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#endif
}

const char* GridSamplerPlugin::getPluginType() const noexcept { return GRID_SAMPLER_PLUGIN_NAME; }

const char* GridSamplerPlugin::getPluginVersion() const noexcept {
  return GRID_SAMPLER_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* GridSamplerPlugin::clone() const noexcept {
  return new GridSamplerPlugin{interpolation_mode_, padding_mode_, align_corners_, data_type_};
}

void GridSamplerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* GridSamplerPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType GridSamplerPlugin::getOutputDataType(int index,
                                                        const nvinfer1::DataType* inputTypes,
                                                        int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void GridSamplerPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc* out,
                                        int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }
}

GridSamplerPluginCreator::GridSamplerPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("interpolation_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("padding_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* GridSamplerPluginCreator::getPluginName() const noexcept {
  return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPluginCreator::getPluginVersion() const noexcept {
  return GRID_SAMPLER_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* GridSamplerPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt* GridSamplerPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  int interpolation_mode{}, padding_mode{}, align_corners{};
  const nvinfer1::PluginField* fields = fc->fields;
  int data_type = 0;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "interpolation_mode")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      interpolation_mode = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "padding_mode")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      padding_mode = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "align_corners")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      align_corners = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj = new GridSamplerPlugin(interpolation_mode, padding_mode, align_corners,
                                   static_cast<nvinfer1::DataType>(data_type));
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* GridSamplerPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  auto* obj = new GridSamplerPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
