#pragma once

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/tf_tensor.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/fwd_utils.h"

// Create Random Integer following the uniform distribution.
inline int RandomInt() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> dis;
  return dis(gen);
}

// Get element size of TF_DataType
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

// return shape of Tensor as a vector
inline std::vector<int64_t> GetTensorShape(const TF_Tensor* tensor) {
  const int num_dim = TF_NumDims(tensor);
  std::vector<int64_t> shape;
  for (int i = 0; i < num_dim; ++i) {
    shape.push_back(TF_Dim(tensor, i));
  }
  return shape;
}

// Create Allocated EmptyTensor with given DataType and dimensions
inline std::shared_ptr<TF_Tensor> CreateEmptyTensor(TF_DataType data_type,
                                                    const std::vector<int64_t>& dims) {
  const size_t size = std::accumulate(dims.begin(), dims.end(), 1ll, std::multiplies<int64_t>());
  const auto data_len = size * GetElementSize(data_type);

  auto tensor = std::shared_ptr<TF_Tensor>(
      TF_AllocateTensor(data_type, dims.data(), static_cast<int>(dims.size()), data_len),
      TF_DeleteTensor);

  assert(tensor != nullptr);
  return tensor;
}

// Create TF_Tensor with given DataType, Dimensions, and data pointer
template <typename T>
std::shared_ptr<TF_Tensor> CreateTensor(TF_DataType data_type, const std::vector<int64_t>& dims,
                                        const T* data) {
  std::shared_ptr<TF_Tensor> tensor = CreateEmptyTensor(data_type, dims);

  assert(tensor != nullptr);
  auto tensor_data = TF_TensorData(tensor.get());
  if (tensor_data == nullptr) return nullptr;

  const auto data_len = TF_TensorByteSize(tensor.get());
  if (data != nullptr && data_len != 0) {
    memcpy(tensor_data, data, data_len);
  }

  return tensor;
}

// Create Random-Value TF_Tensor with given DataType and Dimensions. The range of random_value is
// less than max_val.
template <typename T>
std::shared_ptr<TF_Tensor> CreateRandomTensor(TF_DataType data_type,
                                              const std::vector<int64_t>& dims,
                                              int max_val = 10000) {
  auto size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::int64_t>());

  // TODO(yzx): 根据 data_type 创建，暂时默认全是 float
  std::vector<T> data(size);
  for (auto& item : data) {
    item = RandomInt() % max_val;
  }
  return CreateTensor<T>(data_type, dims, data.data());
}

// Cast TF_Tensor to target TF_Type. The data is deep copied into new TF_Tensor.
template <typename T>
std::shared_ptr<TF_Tensor> CastTensor(const TF_Tensor* tensor, TF_DataType dest_type) {
  assert(sizeof(T) == TF_DataTypeSize(dest_type));

  if (tensor == nullptr) return {};

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
      std::transform(src, src + count, dest, [](uint16_t val) {
        return static_cast<T>(fwd::FwdUtils::Half2FloatFast(val));
      });
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

// return the data of TF_Tensor as a vector
template <typename T>
std::vector<T> GetTensorData(const TF_Tensor* tensor) {
  if (tensor == nullptr) return {};

  auto data = static_cast<T*>(TF_TensorData(tensor));
  auto size = TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor));
  if (data == nullptr || size <= 0) return {};

  return {data, data + size};
}
