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

#include <vector>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/plugin.h"

FWD_TRT_NAMESPACE_BEGIN

constexpr const char* EMBEDDING_BAG_PLUGIN_NAME{"EmbeddingBag_TRT"};
constexpr const char* EMBEDDING_BAG_PLUGIN_VERSION{"001"};

enum class ReduceOperation { GATHER = 999, SUM = 0, MEAN = 1, MAX = 2, SQUARE_SUM = 3 };
void Gather(const int* input, float* output, int batch_size, int offset, const float* data,
            int count, int dim, int data_offset, const cudaStream_t& stream);

void EmbeddingBagWithFixedOffset(const int* input, float* output, int batch_size, int offset,
                                 const float* data, int count, int dim, int data_offset,
                                 int op_type, const cudaStream_t& stream);

void EmbeddingBag(const int* input, const int* input_2, float* output, int input_size,
                  int block_size, const float* data, int count, int dim, int data_offset,
                  int op_type, const cudaStream_t& stream);

class EmbeddingBagPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  /*
   * \param data embedding map data
   * \param dim dimension of embedding vectors
   * \param count number of embedding vectors
   * \param offset the starting index of embedding vectors
   * \param type type of reduce method
   */
  EmbeddingBagPlugin(const float* data, int dim, int count, int offset, ReduceOperation op);

  EmbeddingBagPlugin(void const* serialData, size_t serialLength);

  EmbeddingBagPlugin() = delete;

  ~EmbeddingBagPlugin() noexcept override;

  int getNbOutputs() const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) noexcept override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

  int initialize() noexcept override;

  void terminate() noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void* buffer) const noexcept override;

  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  void destroy() noexcept override;

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;

 private:
  /**
   * \brief 特征维度
   */
  int dim_;

  /**
   * \brief 特征数量
   */
  int count_;

  /**
   * \brief 特征起始id(预留用于可能的多机搜索)
   */
  int data_offset_;

  ReduceOperation op_;

  /**
   * \brief 特征数据
   */
  float* d_data_;
  std::vector<float> h_data_;

  bool initialized_{false};

  const char* mPluginNamespace = "";
};

class EmbeddingBagPluginCreator : public nvinfer1::plugin::BaseCreator {
 public:
  EmbeddingBagPluginCreator();

  ~EmbeddingBagPluginCreator() override = default;

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
