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

#include "trt_engine/trt_network_crt/plugins/upsampler_plugin/upsample_bilinear_2d.h"

#include <cuda_fp16.h>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_UPSAMPLE_BILINEAR_2D_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

UpsampleBilinear2DPlugin::UpsampleBilinear2DPlugin(int output_h, int output_w, int align_corners,
                                                   float scale_h, float scale_w,
                                                   nvinfer1::DataType data_type)
    : output_h_(output_h),
      output_w_(output_w),
      align_corners_(align_corners),
      scale_h_(scale_h),
      scale_w_(scale_w),
      data_type_(data_type),
      initialized_(false) {}

UpsampleBilinear2DPlugin::UpsampleBilinear2DPlugin(void const* serialData, size_t serialLength)
    : initialized_(false) {
  deserialize_value(&serialData, &serialLength, &output_h_);
  deserialize_value(&serialData, &serialLength, &output_w_);
  deserialize_value(&serialData, &serialLength, &align_corners_);
  deserialize_value(&serialData, &serialLength, &scale_h_);
  deserialize_value(&serialData, &serialLength, &scale_w_);
  deserialize_value(&serialData, &serialLength, &data_type_);
}

UpsampleBilinear2DPlugin::~UpsampleBilinear2DPlugin() { terminate(); }

int UpsampleBilinear2DPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs UpsampleBilinear2DPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  ASSERT(nbInputs == 1);
  ASSERT(inputs[0].nbDims == 4);

  nvinfer1::DimsExprs output(inputs[0]);
  output.d[2] = exprBuilder.constant(output_h_);
  output.d[3] = exprBuilder.constant(output_w_);

  return output;
}

int UpsampleBilinear2DPlugin::initialize() noexcept {
  if (initialized_) {
    return 0;
  }

  initialized_ = true;
  return 0;
}

void UpsampleBilinear2DPlugin::terminate() noexcept {
  if (!initialized_) {
    return;
  }

  initialized_ = false;
}

size_t UpsampleBilinear2DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                  int nbInputs,
                                                  const nvinfer1::PluginTensorDesc* outputs,
                                                  int nbOutputs) const noexcept {
  return 0;
}

int UpsampleBilinear2DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                      const nvinfer1::PluginTensorDesc* outputDesc,
                                      const void* const* inputs, void* const* outputs,
                                      void* workspace, cudaStream_t stream) noexcept {
  ASSERT(inputDesc[0].dims.nbDims == 4);
#ifdef ENABLE_UPSAMPLE_BILINEAR_2D_FLOAT16
  switch (data_type_) {
#else
  switch (inputDesc[0].type) {
#endif
    case nvinfer1::DataType::kFLOAT: {
      TensorInfo<float> output_tensor{static_cast<float*>(outputs[0]), outputDesc[0].dims};
      UpSampleBilinear2DCuda<float>({static_cast<const float*>(inputs[0]), inputDesc[0].dims},
                                    output_tensor, output_h_, output_w_, align_corners_, scale_h_,
                                    scale_w_, stream);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      TensorInfo<__half> output_tensor{static_cast<__half*>(outputs[0]), outputDesc[0].dims};
      UpSampleBilinear2DCuda<__half>({static_cast<const __half*>(inputs[0]), inputDesc[0].dims},
                                     output_tensor, output_h_, output_w_, align_corners_, scale_h_,
                                     scale_w_, stream);
      break;
    }
    default: {
      getLogger()->log(nvinfer1::ILogger::Severity::kERROR,
                       "[Upsample Bilinear] Unsupported input data type");
      return -1;
    }
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t UpsampleBilinear2DPlugin::getSerializationSize() const noexcept {
  return serialized_size(output_h_) + serialized_size(output_w_) + serialized_size(align_corners_) +
         serialized_size(scale_h_) + serialized_size(scale_w_) + serialized_size(data_type_);
}

void UpsampleBilinear2DPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, output_h_);
  serialize_value(&buffer, output_w_);
  serialize_value(&buffer, align_corners_);
  serialize_value(&buffer, scale_h_);
  serialize_value(&buffer, scale_w_);
  serialize_value(&buffer, data_type_);
}

bool UpsampleBilinear2DPlugin::supportsFormatCombination(int pos,
                                                         const nvinfer1::PluginTensorDesc* inOut,
                                                         int nbInputs, int nbOutputs) noexcept {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));

#ifdef ENABLE_UPSAMPLE_BILINEAR_2D_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

const char* UpsampleBilinear2DPlugin::getPluginType() const noexcept {
  return UPSAMPLE_BILINEAR_2D_PLUGIN_NAME;
}

const char* UpsampleBilinear2DPlugin::getPluginVersion() const noexcept {
  return UPSAMPLE_BILINEAR_PLUGIN_VERSION;
}

void UpsampleBilinear2DPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* UpsampleBilinear2DPlugin::clone() const noexcept {
  return new UpsampleBilinear2DPlugin{output_h_, output_w_, align_corners_,
                                      scale_h_,  scale_w_,  data_type_};
}

void UpsampleBilinear2DPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* UpsampleBilinear2DPlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace;
}

nvinfer1::DataType UpsampleBilinear2DPlugin::getOutputDataType(int index,
                                                               const nvinfer1::DataType* inputTypes,
                                                               int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void UpsampleBilinear2DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                               int nbInputs,
                                               const nvinfer1::DynamicPluginTensorDesc* out,
                                               int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }
}

UpsampleBilinear2DPluginCreator::UpsampleBilinear2DPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("output_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("output_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("scale_h", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("scale_w", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* UpsampleBilinear2DPluginCreator::getPluginName() const noexcept {
  return UPSAMPLE_BILINEAR_2D_PLUGIN_NAME;
}

const char* UpsampleBilinear2DPluginCreator::getPluginVersion() const noexcept {
  return UPSAMPLE_BILINEAR_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* UpsampleBilinear2DPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt* UpsampleBilinear2DPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  int output_h{}, output_w{}, align_corners{};
  float scale_h{1.0f}, scale_w{1.0f};
  const nvinfer1::PluginField* fields = fc->fields;
  int data_type = 0;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "output_h")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      output_h = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "output_w")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      output_w = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "align_corners")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      align_corners = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "scale_h")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      scale_h = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "scale_w")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      scale_w = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj = new UpsampleBilinear2DPlugin(output_h, output_w, align_corners, scale_h, scale_w,
                                          static_cast<nvinfer1::DataType>(data_type));
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* UpsampleBilinear2DPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  auto* obj = new UpsampleBilinear2DPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
