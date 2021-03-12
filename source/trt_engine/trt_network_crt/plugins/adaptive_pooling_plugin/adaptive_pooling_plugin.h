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

#pragma once

#include <cudnn.h>

#include <string>
#include <vector>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/plugin.h"
#include "trt_engine/trt_network_crt/plugins/common/trt_tensor_info.h"

FWD_TRT_NAMESPACE_BEGIN

constexpr const char* ADAPTIVE_POOLING_PLUGIN_NAME{"AdaptivePooling_TRT"};
constexpr const char* ADAPTIVE_POOLING_PLUGIN_VERSION{"001"};

enum class PoolingOperation { MAX_POOLING = 0, AVG_POOLING };

template <typename T>
void AdaptivePooling2DCuda(const TensorInfo<T>& input, TensorInfo<T>& output,
                           const std::vector<int>& output_size, PoolingOperation type,
                           cudaStream_t stream);

template <typename T>
void AdaptivePooling3DCuda(const TensorInfo<T>& input, TensorInfo<T>& output,
                           const std::vector<int>& output_size, PoolingOperation type,
                           cudaStream_t stream);

class AdaptivePoolingPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  AdaptivePoolingPlugin(const std::vector<int>& output_size, int type,
                        nvinfer1::DataType data_type);

  AdaptivePoolingPlugin(void const* serialData, size_t serialLength);

  AdaptivePoolingPlugin() = delete;

  ~AdaptivePoolingPlugin() override;

  int getNbOutputs() const override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

  int initialize() override;

  void terminate() override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

 private:
  std::vector<int> output_size_;

  int type_;

  nvinfer1::DataType data_type_;

  std::string mPluginNamespace;
};

class AdaptivePoolingPluginCreator : public nvinfer1::plugin::BaseCreator {
 public:
  AdaptivePoolingPluginCreator();

  ~AdaptivePoolingPluginCreator() override = default;

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
};

FWD_TRT_NAMESPACE_END
