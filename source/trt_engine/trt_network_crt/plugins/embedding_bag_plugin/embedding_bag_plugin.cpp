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
//          Zhaoyi LUO (luozy63@gmail.com)

#include "trt_engine/trt_network_crt/plugins/embedding_bag_plugin/embedding_bag_plugin.h"

#include <algorithm>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"
#include "trt_engine/trt_network_crt/plugins/common/trt_tensor_info.h"

FWD_TRT_NAMESPACE_BEGIN

EmbeddingBagPlugin::EmbeddingBagPlugin(const float* data, int dim, int count, int offset,
                                       ReduceOperation op)
    : dim_(dim), count_(count), data_offset_(offset), op_(op), initialized_(false) {
  h_data_.resize(dim_ * count_);
  memcpy(h_data_.data(), data, dim_ * count_ * sizeof(float));
}

EmbeddingBagPlugin::EmbeddingBagPlugin(void const* serialData, size_t serialLength)
    : initialized_(false) {
  deserialize_value(&serialData, &serialLength, &dim_);
  deserialize_value(&serialData, &serialLength, &count_);
  deserialize_value(&serialData, &serialLength, &data_offset_);
  deserialize_value(&serialData, &serialLength, &op_);
  deserialize_value(&serialData, &serialLength, &h_data_);
}

EmbeddingBagPlugin::~EmbeddingBagPlugin() { terminate(); }

int EmbeddingBagPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs EmbeddingBagPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // 单一输入的情形，要求输入是二维Tensor
  if (nbInputs == 1) {
    ASSERT(inputs[0].nbDims == 2);
    if (op_ != ReduceOperation::GATHER) {
      nvinfer1::DimsExprs output = inputs[0];
      output.d[1] = exprBuilder.constant(dim_);
      return output;
    } else {
      nvinfer1::DimsExprs output = inputs[0];
      // This is the workaround to handle tf.gather op. In the case with one signle input with
      // dimensions [1, 3], its output would be reshaped to [1, 3, 1] given dim_ == 1. I'm not clear
      // why there's a special branch for 'ReduceOperation::GATHER' to do so, but that output shape
      // should have been the same as the input shape. To minimize the impact, I only try to extend
      // to the third dimension if dim_ > 1, while it shouldn't be that way either.
      if (dim_ > 1) {
        output.nbDims += 1;
        output.d[2] = exprBuilder.constant(dim_);
      }
      return output;
    }
  }

  // offset一般是一维，但DLRM中是从二维数组中抽取一维做offset
  // 目前的select TensorRT 实现不会为select操作降维
  ASSERT(inputs[1].nbDims == 1 ||
         (inputs[1].nbDims == 2 && inputs[1].d[0]->getConstantValue() == 1));

  nvinfer1::DimsExprs output;

  if (op_ == ReduceOperation::GATHER) {
    // 暂不支持每个batch中抽取数量可能不同的纯提取
    ASSERT(false);
  }

  output.nbDims = 2;

  output.d[0] = inputs[1].nbDims == 1 ? inputs[1].d[0] : inputs[1].d[1];
  output.d[1] = exprBuilder.constant(dim_);

  return output;
}

bool EmbeddingBagPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                   int nbInputs, int nbOutputs) noexcept {
  // 输入为Int 输出为Float(特征向量)
  ASSERT(inOut && nbInputs <= 2 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));

  if (pos < nbInputs) {
    return (inOut[pos].type == nvinfer1::DataType::kINT32 &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  }
}

void EmbeddingBagPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                         const nvinfer1::DynamicPluginTensorDesc* out,
                                         int nbOutputs) noexcept {
  if (initialized_) {
    return;
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data_), dim_ * count_ * sizeof(float)));
  CUDA_CHECK(
      cudaMemcpy(d_data_, h_data_.data(), dim_ * count_ * sizeof(float), cudaMemcpyHostToDevice));
  initialized_ = true;
}

size_t EmbeddingBagPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                            const nvinfer1::PluginTensorDesc* outputs,
                                            int nbOutputs) const noexcept {
  return 0;
}

int EmbeddingBagPlugin::initialize() noexcept { return 0; }

void EmbeddingBagPlugin::terminate() noexcept {
  if (!initialized_) {
    return;
  }

  CUDA_CHECK(cudaFree(d_data_));
  initialized_ = false;
}

int EmbeddingBagPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                const nvinfer1::PluginTensorDesc* outputDesc,
                                const void* const* inputs, void* const* outputs, void* workspace,
                                cudaStream_t stream) noexcept {
  if (inputDesc[0].type == nvinfer1::DataType::kINT32) {
    // 如果输入indices本来就是二维，则是batch_size *
    // input_count的格式，每个batch的offset就是input_count*Batch_number
    if (inputDesc[0].dims.nbDims > 1) {
      const int batch_size = inputDesc[0].dims.d[0];
      const int offset = inputDesc[0].dims.d[1];
      // 纯Gather
      if (op_ == ReduceOperation::GATHER) {
        Gather(static_cast<const int*>(inputs[0]), static_cast<float*>(outputs[0]), batch_size,
               offset, d_data_, count_, dim_, data_offset_, stream);
        CUDA_CHECK(cudaGetLastError());
        return 0;
      }

      // 带Reduce的情形
      EmbeddingBagWithFixedOffset(static_cast<const int*>(inputs[0]),
                                  static_cast<float*>(outputs[0]), batch_size, offset, d_data_,
                                  count_, dim_, data_offset_, static_cast<int>(op_), stream);
      CUDA_CHECK(cudaGetLastError());
      return 0;
    }

    // 输入非二维时，考虑offset数组(对应torch.nn.embedding_bag)
    const int input_size = inputDesc[0].dims.d[0];

    // 每个block处理一组数据, block size = offset的数量
    const int block_size =
        inputDesc[1].dims.nbDims == 1 ? inputDesc[1].dims.d[0] : inputDesc[1].dims.d[1];

    EmbeddingBag(static_cast<const int*>(inputs[0]), static_cast<const int*>(inputs[1]),
                 static_cast<float*>(outputs[0]), input_size, block_size, d_data_, count_, dim_,
                 data_offset_, static_cast<int>(op_), stream);
  } else {
    getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Unsupported input data type");
    return -1;
  }
  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t EmbeddingBagPlugin::getSerializationSize() const noexcept {
  return serialized_size(dim_) + serialized_size(count_) + serialized_size(data_offset_) +
         serialized_size(op_) + serialized_size(h_data_);
}

void EmbeddingBagPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, dim_);
  serialize_value(&buffer, count_);
  serialize_value(&buffer, data_offset_);
  serialize_value(&buffer, op_);
  serialize_value(&buffer, h_data_);
}

const char* EmbeddingBagPlugin::getPluginType() const noexcept { return EMBEDDING_BAG_PLUGIN_NAME; }

const char* EmbeddingBagPlugin::getPluginVersion() const noexcept {
  return EMBEDDING_BAG_PLUGIN_VERSION;
}

void EmbeddingBagPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* EmbeddingBagPlugin::clone() const noexcept {
  return new EmbeddingBagPlugin(h_data_.data(), dim_, count_, data_offset_, op_);
}

void EmbeddingBagPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* EmbeddingBagPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType EmbeddingBagPlugin::getOutputDataType(int index,
                                                         const nvinfer1::DataType* inputTypes,
                                                         int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return nvinfer1::DataType::kFLOAT;
}

EmbeddingBagPluginCreator::EmbeddingBagPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("count", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("offset", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("op", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* EmbeddingBagPluginCreator::getPluginName() const noexcept {
  return EMBEDDING_BAG_PLUGIN_NAME;
}

const char* EmbeddingBagPluginCreator::getPluginVersion() const noexcept {
  return EMBEDDING_BAG_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* EmbeddingBagPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt* EmbeddingBagPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  int dim;
  int count;
  int offset;
  ReduceOperation op;
  const float* data;
  int type = 0;
  const nvinfer1::PluginField* fields = fc->fields;
  ASSERT(fc->nbFields == 5);
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "dim")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      dim = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "count")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      count = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "offset")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      offset = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "op")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      op = static_cast<ReduceOperation>(*(static_cast<const int*>(fields[i].data)));
    } else if (!strcmp(attrName, "data")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      data = static_cast<const float*>(fields[i].data);
    } else {
      ASSERT(false);
    }
  }

  auto obj = new EmbeddingBagPlugin(data, dim, count, offset, op);
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* EmbeddingBagPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  auto* obj = new EmbeddingBagPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
