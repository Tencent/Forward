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

constexpr const char* GRID_SAMPLER_PLUGIN_NAME{"GridSampler_TRT"};
constexpr const char* GRID_SAMPLER_PLUGIN_VERSION{"001"};

template <typename T>
void GridSampler2DCuda(const TensorInfo<T>& input, const TensorInfo<T>& grid, TensorInfo<T>& output,
                       int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                       cudaStream_t stream);

template <typename T>
void GridSampler3DCuda(const TensorInfo<T>& input, const TensorInfo<T>& grid, TensorInfo<T>& output,
                       int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
                       cudaStream_t stream);

class GridSamplerPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  GridSamplerPlugin(int interpolation_mode, int padding_mode, int align_corners,
                    nvinfer1::DataType data_type);

  GridSamplerPlugin(void const* serialData, size_t serialLength);

  GridSamplerPlugin() = delete;

  ~GridSamplerPlugin() noexcept override;

  int getNbOutputs() const noexcept override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  int initialize() noexcept override;

  void terminate() noexcept override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void* buffer) const noexcept override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) noexcept override;

  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  void destroy() noexcept override;

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

 private:
  int interpolation_mode_;

  int padding_mode_;

  int align_corners_;

  nvinfer1::DataType data_type_;

  const char* mPluginNamespace = "";
};

class GridSamplerPluginCreator : public nvinfer1::plugin::BaseCreator {
 public:
  GridSamplerPluginCreator();

  ~GridSamplerPluginCreator() override = default;

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,
                                              const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
                                                   size_t serialLength) noexcept override;

 private:
  nvinfer1::PluginFieldCollection mFC;

  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

FWD_TRT_NAMESPACE_END
