/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <vector>

#include "trt_engine/trt_network_crt/plugins/common/plugin_common.h"

FWD_TRT_NAMESPACE_BEGIN

// Clip plugin specific constants
static const char* FWD_LAYER_NORM_VERSION{"1"};
static const char* FWD_LAYER_NORM_NAME{"ForwardLayerNormPluginDynamic"};

template <bool hasBias>
int computeLayerNormDQQ(cudaStream_t stream, const int ld, const int n, const int8_t* input,
                        const __half* beta, const __half* gamma, int8_t* output, const __half* bias,
                        const float dqScaleIn, const float qScale);

template <typename T, bool hasBias>
int computeLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* beta,
                     const T* gamma, T* output, const T* bias);

class LayerNormPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  LayerNormPluginDynamic(const std::string name, const nvinfer1::DataType type, const int ld,
                         const nvinfer1::Weights& beta, const nvinfer1::Weights& gamma,
                         const nvinfer1::Weights& bias);

  LayerNormPluginDynamic(const std::string name, const void* data, size_t length);

  // It doesn't make sense to make LayerNormPluginDynamic without arguments,
  // so we delete default constructor.
  LayerNormPluginDynamic() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  bool mHasBias;
  nvinfer1::DataType mType;
  nvinfer1::DataType mCfgType;
  size_t mParamWordsize;
  size_t mLd;  // leading dim

  const std::string mLayerName;
  std::string mNamespace;

  fwd::trt_::cuda_unique_ptr<void> mZerosDev;

  fwd::trt_::cuda_unique_ptr<void> mGammaDev;
  fwd::trt_::cuda_unique_ptr<void> mBetaDev;
  fwd::trt_::WeightsWithOwnership mGamma;
  fwd::trt_::WeightsWithOwnership mBeta;
  fwd::trt_::cuda_unique_ptr<void> mBiasDev;
  fwd::trt_::WeightsWithOwnership mBias;
};

class LayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  LayerNormPluginDynamicCreator();

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

 private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

FWD_TRT_NAMESPACE_END
