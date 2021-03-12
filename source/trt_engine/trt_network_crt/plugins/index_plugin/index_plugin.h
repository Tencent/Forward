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

constexpr const char* INDEX_PLUGIN_NAME{"Index_TRT"};
constexpr const char* INDEX_PLUGIN_VERSION{"001"};

void IndexCuda(const float* input, float* output, int* index, int* input_dims, int* output_dims,
               int* index_pos, int nb_input_dims, int nb_output_dims, int nb_index, bool put_front,
               const cudaStream_t& stream);

class IndexPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  /*
   * \param data embedding map data
   * \param dim dimension of embedding vectors
   * \param count number of embedding vectors
   * \param offset the starting index of embedding vectors
   * \param type type of reduce method
   */
  IndexPlugin(const int* index_data, const int* index_dims, int dim_count, int index_dim_count,
              int index_count);

  IndexPlugin(void const* serialData, size_t serialLength);

  IndexPlugin() = delete;

  ~IndexPlugin() override;

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
  bool CheckIndexPosition(int* index_dims);

  /**
   * \brief 维度数量
   */
  int nb_dims_;

  /**
   * \brief 取下标索引的维度数量
   */
  int nb_index_dims_;

  /**
   * \brief 索引数量
   */
  int nb_index_;

  /**
   * \brief 输入输出的维度
   */
  int* d_input_dims_;
  int* d_output_dims_;

  /**
   * \brief 下标数据
   */
  int* d_data_;
  std::vector<int> h_data_;

  /**
   * \brief 各个位置是否取索引。0为不取，1为取
   */
  int* d_pos_;
  std::vector<int> h_pos_;

  /**
   * \brief 索引列是否前置
   */
  bool put_in_the_front_;

  bool initialized_;

  const char* mPluginNamespace = "";

  std::vector<nvinfer1::Dims> input_dims_;
};

class IndexPluginCreator : public nvinfer1::plugin::BaseCreator {
 public:
  IndexPluginCreator();

  ~IndexPluginCreator() override = default;

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
