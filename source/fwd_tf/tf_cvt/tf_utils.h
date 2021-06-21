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

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/fwd_common.h"
#include "common/fwd_utils.h"
#include "common/fwd_weights.h"
#include "common/trt_utils.h"
#include "fwd_tf/tf_cvt/tf_cpp_api.h"

#ifdef _MSC_VER
#undef max
#undef min
#endif

FWD_TF_NAMESPACE_BEGIN

/**
 * \brief 生成 0-1 之间的均匀分布随机浮点数
 * \return
 */
float RandomFloat();

/**
 * \brief 生成 0-INT_MAX 之间的均匀分布随机整数
 * \return
 */
int RandomInt();

std::shared_ptr<TF_Tensor> CreateEmptyTensor(TF_DataType data_type,
                                             const std::vector<int64_t>& dims);

/**
 * \brief 将 vector 数据封装成一个 TF_Tensor
 * \tparam T 数值类型
 * \param data_type TF 数值类型
 * \param dims TF_Tensor 维度
 * \param data Vector 数据
 * \return
 */
template <typename T>
std::shared_ptr<TF_Tensor> CreateTensor(TF_DataType data_type, const std::vector<int64_t>& dims,
                                        const T* data) {
  std::shared_ptr<TF_Tensor> tensor = CreateEmptyTensor(data_type, dims);

  auto tensor_data = TF_TensorData(tensor.get());
  if (tensor_data == nullptr) return nullptr;

  const auto data_len = TF_TensorByteSize(tensor.get());
  if (data != nullptr && data_len != 0) {
    memcpy(tensor_data, data, data_len);
  }

  return tensor;
}

/**
 * \brief 随机创建一个 TF_Tensor
 * \tparam T 数值类型
 * \param data_type TF 数值类型
 * \param dims TF_Tensor 维度
 * \return std::shared_ptr<TF_Tensor>
 */
template <typename T>
std::shared_ptr<TF_Tensor> CreateRandomTensor(TF_DataType data_type,
                                              const std::vector<int64_t>& dims) {
  auto size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::int64_t>());

  // TODO(yzx): 根据 data_type 创建，暂时默认全是 float
  std::vector<T> data(size);
  for (auto& item : data) {
    item = RandomFloat();
  }
  return CreateTensor<T>(data_type, dims, data.data());
}

template <typename T>
std::shared_ptr<TF_Tensor> CreateRandomIntTensor(TF_DataType data_type,
                                                 const std::vector<int64_t>& dims,
                                                 int max = 10000) {
  auto size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::int64_t>());

  std::vector<T> data(size);
  for (auto& item : data) {
    item = RandomInt() % max;
  }
  return CreateTensor<T>(data_type, dims, data.data());
}

/**
 * \brief 为 TF_Graph 创建 TF_Session 会话
 * \param graph TF_Graph
 * \param options 会话可选项
 * \return std::shared_ptr<TF_Session>
 */
std::shared_ptr<TF_Session> CreateSession(const Graph& graph,
                                          std::shared_ptr<TF_SessionOptions> options = nullptr);

inline std::vector<int64_t> GetTensorShape(const TF_Tensor* tensor) {
  const int num_dim = TF_NumDims(tensor);
  std::vector<int64_t> shape;
  for (int i = 0; i < num_dim; ++i) {
    shape.push_back(TF_Dim(tensor, i));
  }
  return shape;
}

inline size_t GetElementSize(TF_DataType type) {
  switch (type) {
    case TF_INT64:
      return 8;
    case TF_INT32:
      return 4;
    case TF_INT16:
      return 2;
    case TF_INT8:
      return 1;
    case TF_FLOAT:
      return 4;
    case TF_HALF:
      return 2;
    default:
      return 0;
  }
}

/**
 * \brief 释放 TF_Buffer 的资源
 * \param data
 */
inline void DeallocateBuffer(void* data, size_t) { std::free(data); }

/**
 * \brief 删除 TF_Session 会话
 * \param session 会话
 * \return
 */
TF_Code DeleteSession(TF_Session* session);

/**
 * \brief 获取 TF_Tensor 的维度
 * \param tensor
 * \return nvinfer1::Dims
 */
inline nvinfer1::Dims DimsOf(const TF_Tensor* tensor) {
  nvinfer1::Dims dims{-1};
  dims.nbDims = TF_NumDims(tensor);
  for (int i = 0; i < dims.nbDims; i++) {
    dims.d[i] = TF_Dim(tensor, i);
  }
  return dims;
}

inline nvinfer1::Dims DimsOf(const Output& output) {
  Status status;

  int num_dims = output.GetTensorNumDims();
  if (num_dims <= 0) return {};

  std::vector<int64_t> dims(num_dims);
  output.GetTensorShape(dims.data(), num_dims);

  nvinfer1::Dims nv_dims{-1};
  nv_dims.nbDims = num_dims;
  for (int i = 0; i < dims.size(); ++i) {
    nv_dims.d[i] = dims[i];
  }

  return nv_dims;
}

inline nvinfer1::Dims DimsOf(const Operation& op) { return DimsOf(Output(op.Graph(), op.Op(), 0)); }

/**
 * \brief 从 TF_Tensor 中取出数据转换为 Vector 存储
 * \tparam T 数值类型
 * \param tensor TF_Tensor
 * \return Vector<T>
 */
template <typename T>
std::vector<T> GetTensorData(const TF_Tensor* tensor) {
  if (tensor == nullptr) {
    return {};
  }
  auto data = static_cast<T*>(TF_TensorData(tensor));
  auto size = TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor));
  if (data == nullptr || size <= 0) {
    return {};
  }

  return {data, data + size};
}

template <typename T>
std::shared_ptr<TF_Tensor> CastTensor(const TF_Tensor* tensor, TF_DataType dest_type) {
  CHECK(sizeof(T) == TF_DataTypeSize(dest_type));

  if (tensor == nullptr) {
    return {};
  }

  const auto src_type = TF_TensorType(tensor);
  const auto src_data = TF_TensorData(tensor);

  const auto count = TF_TensorElementCount(tensor);

  auto dest_tensor = CreateEmptyTensor(dest_type, GetTensorShape(tensor));
  T* dest = reinterpret_cast<T*>(TF_TensorData(dest_tensor.get()));

  switch (src_type) {
    case TF_FLOAT: {
      const float* src = reinterpret_cast<const float*>(src_data);
      std::transform(src, src + count, dest, [](float val) { return static_cast<T>(val); });
      break;
    }
    case TF_HALF: {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(src_data);
      std::transform(src, src + count, dest,
                     [](uint16_t val) { return static_cast<T>(FwdUtils::Half2FloatFast(val)); });
      break;
    }
    case TF_INT64: {
      const int64_t* src = reinterpret_cast<const int64_t*>(src_data);
      std::transform(src, src + count, dest, [](int64_t val) { return static_cast<T>(val); });
      break;
    }
    case TF_INT32: {
      const int* src = reinterpret_cast<const int*>(src_data);
      std::transform(src, src + count, dest, [](int val) { return static_cast<T>(val); });
      break;
    }
    case TF_INT16: {
      const int16_t* src = reinterpret_cast<const int16_t*>(src_data);
      std::transform(src, src + count, dest, [](int16_t val) { return static_cast<T>(val); });
      break;
    }
    case TF_INT8: {
      const char* src = reinterpret_cast<const char*>(src_data);
      std::transform(src, src + count, dest, [](char val) { return static_cast<T>(val); });
      break;
    }
    default:
      return {};
  }

  return dest_tensor;
}

/**
 * \brief 从 Vector<TF_Tensor*> 中取出数据转换为 嵌套 Vector 存储
 * \tparam T 数值类型
 * \param tensors Vector<TF_Tensor*>
 * \return Vector<Vector<T>>
 */
template <typename T>
std::vector<std::vector<T>> GetTensorsData(const std::vector<TF_Tensor*>& tensors) {
  std::vector<std::vector<T>> data;
  data.reserve(tensors.size());
  for (auto t : tensors) {
    data.push_back(GetTensorData<T>(t));
  }
  return data;
}

/**
 * \brief 从 Vector<shared_ptr<TF_Tensor>> 中取出数据转换为 嵌套 Vector 存储
 * \tparam T 数值类型
 * \param tensors Vector<shared_ptr<TF_Tensor>>
 * \return Vector<Vector<T>>
 */
template <typename T>
std::vector<std::vector<T>> GetTensorsData(const std::vector<std::shared_ptr<TF_Tensor>>& tensors) {
  std::vector<std::vector<T>> data;
  data.reserve(tensors.size());
  for (auto t : tensors) {
    data.push_back(GetTensorData<T>(t.get()));
  }
  return data;
}

/**
 * \brief 从文件读取数据缓存到 TF_Buffer
 * \param filename 文件路径
 * \return
 */
TF_Buffer* ReadBufferFromFile(const std::string& filename);

/**
 * \brief 执行 TF_Session 会话
 * \param session 会话
 * \param inputs 输入节点
 * \param input_tensors 输入数据 Tensors
 * \param ninputs 输入的数量
 * \param outputs 输出节点
 * \param output_tensors 输出数据 Tensors
 * \param noutputs 输出的数量
 * \return 执行状态码 TF_Code
 */
inline bool RunSession(TF_Session* session, const TF_Output* inputs,
                       TF_Tensor* const* input_tensors, std::size_t ninputs,
                       const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                       Status& status) {
  if (session == nullptr || inputs == nullptr || input_tensors == nullptr || outputs == nullptr ||
      output_tensors == nullptr) {
    return false;
  }

  TF_SessionRun(session,
                nullptr,  // Run options.
                inputs, input_tensors, static_cast<int>(ninputs),
                // Input tensors, input tensor values, number of inputs.
                outputs, output_tensors, static_cast<int>(noutputs),
                // Output tensors, output tensor values, number of outputs.
                nullptr, 0,  // Target operations, number of targets.
                nullptr,     // Run metadata.
                status);     // Output status.

  return status.Ok();
}

/**
 * \brief 将 TF_Tensor 转换成 FwdWeights
 * \param tf_tensor TF_Tensor 数据
 * \return FwdWeights 数据
 */
inline FwdWeights ToFwdWeights(TF_Tensor* tf_tensor) {
  FwdWeights weights;

  if (tf_tensor == nullptr) {
    return weights;
  }

  int factor = 1;

  const auto data_type = TF_TensorType(tf_tensor);
  // TODO(yzx): Doesn't handle the whole range of TF_Datatype
  // enumeration used (kINT64)
  switch (data_type) {
    case TF_DataType::TF_FLOAT:
      weights.SetType(nvinfer1::DataType::kFLOAT);
      factor = 4;
      break;
    case TF_DataType::TF_HALF:
      weights.SetType(nvinfer1::DataType::kHALF);
      factor = 2;
      break;
    case TF_DataType::TF_INT8:
      weights.SetType(nvinfer1::DataType::kINT8);
      break;
    case TF_DataType::TF_INT32:
      weights.SetType(nvinfer1::DataType::kINT32);
      factor = 4;
      break;
  }

  weights.SetDims(DimsOf(tf_tensor));
  weights.SetCount(TF_TensorElementCount(tf_tensor));

  const char* data_ptr = static_cast<const char*>(TF_TensorData(tf_tensor));
  weights.SetData(data_ptr, weights.Count() * factor);

  return weights;
}

inline FwdWeights ToFwdWeights(const Tensor& tensor) { return ToFwdWeights(tensor.get()); }

inline TF_DataType TfDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT:
      return TF_FLOAT;
    case DataType::HALF:
      return TF_HALF;
    case DataType::INT8:
      return TF_INT8;
    case DataType::INT16:
      return TF_INT16;
    case DataType::INT32:
      return TF_INT32;
    case DataType::DOUBLE:
      return TF_DOUBLE;
    case DataType::INT64:
      return TF_INT64;
    default:
      throw std::runtime_error("Cannot parse fwd::DataType::INVALID to a valid TF_TYPE");
      return TF_FLOAT;
  }
}

template <>
inline std::shared_ptr<TF_Tensor> CastTensor<uint16_t>(const TF_Tensor* tensor,
                                                       TF_DataType dest_type) {
  CHECK_EQ(dest_type, TF_HALF);
  if (tensor == nullptr) {
    return {};
  }

  const auto src_type = TF_TensorType(tensor);
  const auto src = TF_TensorData(tensor);
  const auto ele_size = TF_TensorElementCount(tensor);

  auto dest_tensor = CreateEmptyTensor(dest_type, GetTensorShape(tensor));
  uint16_t* dest = reinterpret_cast<uint16_t*>(TF_TensorData(dest_tensor.get()));

  switch (src_type) {
    case TF_FLOAT:
      std::transform(reinterpret_cast<float*>(src), reinterpret_cast<float*>(src) + ele_size, dest,
                     [](float val) { return FwdUtils::Float2Half(val); });
      break;
    case TF_HALF:
      std::transform(reinterpret_cast<uint16_t*>(src), reinterpret_cast<uint16_t*>(src) + ele_size,
                     dest, [](uint16_t val) { return val; });
      break;
    case TF_INT64:
      std::transform(reinterpret_cast<int64_t*>(src), reinterpret_cast<int64_t*>(src) + ele_size,
                     dest, [](int64_t val) { return FwdUtils::Float2Half(val); });
      break;
    case TF_INT32:
      std::transform(reinterpret_cast<int*>(src), reinterpret_cast<int*>(src) + ele_size, dest,
                     [](int val) { return FwdUtils::Float2Half(val); });
      break;
    case TF_INT16:
      std::transform(reinterpret_cast<int16_t*>(src), reinterpret_cast<int16_t*>(src) + ele_size,
                     dest, [](int16_t val) { return FwdUtils::Float2Half(val); });
      break;
    case TF_INT8:
      std::transform(reinterpret_cast<char*>(src), reinterpret_cast<char*>(src) + ele_size, dest,
                     [](char val) { return FwdUtils::Float2Half(val); });
      break;
    default:
      return {};
  }

  return dest_tensor;
}

FWD_TF_NAMESPACE_END
