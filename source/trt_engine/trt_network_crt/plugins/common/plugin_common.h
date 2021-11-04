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
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <easylogging++.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/plugin.h"

#define TRT_UNUSED (void)

using half = __half;

constexpr uint32_t BDIM = 1;  // batch dimension
constexpr uint32_t SDIM = 0;  // seq len dimension
constexpr uint32_t HDIM = 2;  // hidden dimension

constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;

// For full mask mode, we must produce the compressed mask format expected by
// the fused attention path. Currently, only two sequence lengths are supported.
// We hard code the sizes here. The number of threads per CTA: warps_m * warps_n
// * warps_k * 32;
constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;

// The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M
// dimension: (s + 16*warps_m - 1) / (16*warps_m);
constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM384 = 24;

// Packed mask size per batch. Layout is XMMAS_M * THREADS_PER_CTA.
constexpr size_t unfusedMaskSize = 1;
constexpr size_t packedMaskSize64 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize96 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;

FWD_TRT_NAMESPACE_BEGIN

inline int getSMVersion() {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  return props.major * 10 + props.minor;
}

inline int getMHAMaskPackedSize(int smVersion, nvinfer1::DataType dataType, int sequenceLength) {
  // this code must match EmbLayerNormPluginDynamic::getOutputDimensions in
  // embLayerNormPlugin.cpp
  int packedSize = unfusedMaskSize;
  if ((smVersion == kSM_75 || smVersion == kSM_80 || smVersion == kSM_86) &&
      (dataType == nvinfer1::DataType::kINT8 || dataType == nvinfer1::DataType::kHALF)) {
    if (sequenceLength == 64) {
      packedSize = (dataType == nvinfer1::DataType::kHALF ? packedMaskSize64 : packedSize);
    } else if (sequenceLength == 96) {
      packedSize = (dataType == nvinfer1::DataType::kHALF ? packedMaskSize96 : packedSize);
    } else if (sequenceLength == 128) {
      packedSize = packedMaskSize128;
    } else if (sequenceLength == 384) {
      packedSize = packedMaskSize384;
    }
  }
  return packedSize;
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
#if NV_TENSORRT_MAJOR >= 7
    case nvinfer1::DataType::kBOOL:
#endif  // NV_TENSORRT_MAJOR >= 7
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
}

inline int64_t getWeightsSize(const nvinfer1::Weights& w, nvinfer1::DataType type) {
  return w.count * getElementSize(type);
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename IntType>
constexpr IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}
template <typename IntType>
constexpr IntType alignTo(IntType a, IntType b) {
  return ceildiv(a, b) * b;
}

template <typename T>
inline T* deserToDev(const char*& buffer, size_t nbElem) {
  void* dev{nullptr};
  const size_t len = sizeof(T) * nbElem;
  CUASSERT(cudaMalloc(&dev, len));
  CUASSERT(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

  buffer += len;
  return static_cast<T*>(dev);
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem) {
  const size_t len = sizeof(T) * nbElem;
  CUASSERT(cudaMemcpy(buffer, static_cast<const void*>(data), len, cudaMemcpyDeviceToHost));
  buffer += len;
}

template <typename T>
inline T* devToDev(const T* data, size_t nbElem) {
  void* dev{nullptr};
  const size_t len = sizeof(T) * nbElem;
  CUASSERT(cudaMalloc(&dev, len));
  CUASSERT(cudaMemcpy(dev, static_cast<const void*>(data), len, cudaMemcpyDeviceToDevice));
  return static_cast<T*>(dev);
}

template <typename T>
struct CudaDeleter {
  void operator()(T* buf) { CUASSERT(cudaFree(buf)); }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, fwd::trt_::CudaDeleter<T>>;

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem) {
  ptr.reset(static_cast<T*>(cudaMem), fwd::trt_::CudaDeleter<T>());
}

struct WeightsWithOwnership : public nvinfer1::Weights {
  WeightsWithOwnership() {
    values = nullptr;
    count = 0;
  }
  ~WeightsWithOwnership() { operator delete[](const_cast<void*>(values)); }

  WeightsWithOwnership(const WeightsWithOwnership&) = delete;
  WeightsWithOwnership operator=(const WeightsWithOwnership&) = delete;
  WeightsWithOwnership(const WeightsWithOwnership&&) = delete;
  WeightsWithOwnership operator=(const WeightsWithOwnership&&) = delete;

  void convertAndCopy(const nvinfer1::Weights& src, nvinfer1::DataType type) {
    this->type = type;
    this->count = src.count;

    if (type == nvinfer1::DataType::kFLOAT) {
      auto destBuf = new float[src.count];
      this->values = destBuf;

      if (src.type == nvinfer1::DataType::kFLOAT) {
        LOG(INFO) << "Float Weights(Host) => Float Array(Host)\n";
        std::copy_n(static_cast<const float*>(src.values), src.count, destBuf);
      } else {
        assert(src.type == nvinfer1::DataType::kHALF);

        LOG(INFO) << "Half Weights(Host) => Float Array(Host)\n";
        const auto s = static_cast<const half*>(src.values);
        auto d = static_cast<float*>(const_cast<void*>(this->values));

        for (auto it = 0; it < src.count; it++) {
          d[it] = __half2float(s[it]);
        }
      }
    } else if (type == nvinfer1::DataType::kHALF) {
      auto destBuf = new half[src.count];
      this->values = destBuf;

      if (src.type == nvinfer1::DataType::kHALF) {
        LOG(INFO) << "Half Weights(Host) => Half Array(Host)\n";
        std::copy_n(static_cast<const half*>(src.values), src.count, destBuf);
      } else {
        assert(src.type == nvinfer1::DataType::kFLOAT);

        LOG(INFO) << "Float Weights(Host) => Half Array(Host)\n";
        const auto s = static_cast<const float*>(src.values);
        auto d = static_cast<half*>(const_cast<void*>(this->values));

        for (auto it = 0; it < src.count; it++) {
          d[it] = __float2half(s[it]);
        }
      }
    } else {
      throw std::runtime_error("Unsupported DataType specified for plugin.");
    }
  }

  void convertAndCopy(const char*& srcBuf, size_t count, nvinfer1::DataType type) {
    this->type = type;
    this->count = count;
    const auto nbBytes = getWeightsSize(*this, type);
    auto destBuf = new char[nbBytes];
    this->values = destBuf;

    std::copy_n(srcBuf, nbBytes, destBuf);
    srcBuf += nbBytes;
  }
};

template <typename T>
inline void copyToDevice(WeightsWithOwnership& hostWeights, size_t nbBytes,
                         cuda_unique_ptr<T>& cudaWeights) {
  if (hostWeights.values) {
    void* cudaMem{nullptr};
    CUASSERT(cudaMalloc(&cudaMem, nbBytes));
    CUASSERT(cudaMemcpy(cudaMem, hostWeights.values, nbBytes, cudaMemcpyHostToDevice));
    cudaWeights.reset(static_cast<T*>(cudaMem));
  }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, float* destDev) {
  size_t wordSize = sizeof(float);
  size_t nbBytes = src.count * wordSize;
  if (src.type == nvinfer1::DataType::kFLOAT) {
    LOG(INFO) << "Float Weights(Host) => Float Array(Device)" << std::endl;
    CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
  } else {
    LOG(INFO) << "Half Weights(Host) => Float Array(Device)" << std::endl;
    std::vector<float> tmp(src.count);
    const half* values = reinterpret_cast<const half*>(src.values);

    for (size_t it = 0; it < tmp.size(); it++) {
      tmp[it] = __half2float(values[it]);
    }

    CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
  }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, half* destDev) {
  size_t wordSize = sizeof(half);
  size_t nbBytes = src.count * wordSize;
  if (src.type == nvinfer1::DataType::kHALF) {
    LOG(INFO) << "Half Weights(Host) => Half Array(Device)" << std::endl;
    CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
  } else {
    LOG(INFO) << "Float Weights(Host) => Half Array(Device)" << std::endl;
    std::vector<half> tmp(src.count);
    const float* values = reinterpret_cast<const float*>(src.values);

    for (size_t it = 0; it < tmp.size(); it++) {
      tmp[it] = __float2half(values[it]);
    }
    CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
  }
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype) {
  switch (ftype) {
    case nvinfer1::PluginFieldType::kFLOAT32: {
      LOG(INFO) << "PluginFieldType is Float32" << std::endl;
      return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16: {
      LOG(INFO) << "PluginFieldType is Float16" << std::endl;
      return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32: {
      LOG(INFO) << "PluginFieldType is Int32" << std::endl;
      return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8: {
      LOG(INFO) << "PluginFieldType is Int8" << std::endl;
      return nvinfer1::DataType::kINT8;
    }
    default:
      throw std::invalid_argument("No corresponding datatype for plugin field type");
  }
}

FWD_TRT_NAMESPACE_END
