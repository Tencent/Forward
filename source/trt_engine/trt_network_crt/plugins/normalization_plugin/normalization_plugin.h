/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cudnn.h>

#include <string>
#include <vector>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/plugin.h"

FWD_NAMESPACE_BEGIN
enum class TrtNormalizationType;

FWD_NAMESPACE_END

FWD_TRT_NAMESPACE_BEGIN

constexpr const char* NORMALIZATION_PLUGIN_VERSION{"001"};
constexpr const char* NORMALIZATION_PLUGIN_NAME{"Normalization_TRT"};

class NormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  NormalizationPlugin(TrtNormalizationType type, float epsilon, const std::vector<float>& scale,
                      const std::vector<float>& bias, nvinfer1::DataType data_type,
                      int max_batch_size);

  NormalizationPlugin(TrtNormalizationType type, float epsilon, const std::vector<float>& scale,
                      const std::vector<float>& bias, const std::vector<float>& mean,
                      const std::vector<float>& var, nvinfer1::DataType data_type,
                      int max_batch_size);

  NormalizationPlugin(void const* serialData, size_t serialLength);

  NormalizationPlugin() = delete;

  ~NormalizationPlugin() override;

  int getNbOutputs() const override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) override;

  int initialize() override;

  void terminate() override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

 private:
  TrtNormalizationType _type;
  float _epsilon;
  int _nchan;
  int max_batch_size_;
  std::vector<float> _h_scale;
  std::vector<float> _h_bias;
  std::vector<float> _h_mean;
  std::vector<float> _h_var;
  float* _d_scale;
  float* _d_bias;
  float* _d_mean;
  float* _d_var;
  nvinfer1::DataType _data_type;
  bool _initialized;
  cudnnHandle_t _cudnn_handle;
  cudnnTensorDescriptor_t _x_desc, _b_desc;
  const char* mPluginNamespace = "";
};

class NormalizationPluginCreator : public nvinfer1::plugin::BaseCreator {
 public:
  NormalizationPluginCreator();

  ~NormalizationPluginCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection* getFieldNames() override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,
                                              const nvinfer1::PluginFieldCollection* fc) override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
                                                   size_t serialLength) override;

 private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace{};
};

FWD_TRT_NAMESPACE_END
