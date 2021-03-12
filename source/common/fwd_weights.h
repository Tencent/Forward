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

#include <NvInfer.h>
#include <string.h>

#include <algorithm>
#include <bitset>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN

// Refer to: TensorRT/rtEx/Utils.cpp::ReshapeWeights
static void Increment(int* stride, const int* shape, const int* order, int nbDims) {
  int count{nbDims - 1};
  do {
    if (++stride[order[count]] < shape[order[count]]) {
      return;
    }
    stride[order[count--]] = 0;
  } while (count >= 0);
}

// Refer to: TensorRT/rtEx/Utils.cpp::ReshapeWeights
static int Compute(const int* stride, const int* coord, int start) {
  int dim{coord[start - 1]};
  while (--start > 0) {
    dim = coord[start - 1] * stride[start - 1] + dim;
  }
  return dim;
}

// Refer to: TensorRT/rtEx/Utils.cpp::ReshapeWeights
// According to the given data TYPE, reshape/transpose the weights inplace.
template <typename TYPE>
static bool ReshapeWeightsInternal(const TYPE* input, const int* shape, const int* order, TYPE* ptr,
                                   int nbDims) {
  int shapeVolume = std::accumulate(shape, shape + nbDims, 1, std::multiplies<int>());
  std::vector<int> outShape(nbDims), linear(nbDims);
  for (int x = 0; x < nbDims; ++x) {
    outShape[x] = shape[order[x]];
    linear[x] = x;
  }
  int inStride[nvinfer1::Dims::MAX_DIMS], inIdx[nvinfer1::Dims::MAX_DIMS];
  int outStride[nvinfer1::Dims::MAX_DIMS], outIdx[nvinfer1::Dims::MAX_DIMS];
  std::fill(inIdx, inIdx + nvinfer1::Dims::MAX_DIMS, 0);
  std::fill(outIdx, outIdx + nvinfer1::Dims::MAX_DIMS, 0);
  inStride[nbDims - 1] = outStride[nbDims - 1] = 1;
  for (int x = nbDims - 1; x > 0; --x) {
    inStride[x - 1] = (inStride[x] * shape[x]);
    outStride[x - 1] = (outStride[x] * outShape[x]);
  }
  for (int x = 0; x < shapeVolume; ++x) {
    auto iIdx = Compute(inStride, inIdx, nbDims);
    auto oIdx = Compute(outStride, outIdx, nbDims);
    ptr[oIdx] = input[iIdx];
    Increment(inIdx, shape, order, nbDims);
    Increment(outIdx, outShape.data(), linear.data(), nbDims);
  }
  return true;
}

// Refer to: TensorRT/rtEx/Utils.cpp::ReshapeWeights
// Reshape/Transpose nvinfer1::Weights
static bool ReshapeWeights(const nvinfer1::Weights& input, const int* shape, const int* order,
                           void* data, int nbDims) {
  // Do comprehensive error checking on inputs
  if (order == nullptr || shape == nullptr || data == nullptr || nbDims == 0 ||
      nbDims > nvinfer1::Dims::MAX_DIMS || input.values == nullptr || input.count == 0 ||
      static_cast<unsigned>(nvinfer1::EnumMax<nvinfer1::DataType>()) <=
          static_cast<unsigned>(input.type)) {
    return false;
  }

  const int* oEnd = order + nbDims;
  int shapeVolume = std::accumulate(shape, shape + nbDims, 1, std::multiplies<int>());
  std::bitset<nvinfer1::Dims::MAX_DIMS> orderIndexMissing((1 << nbDims) - 1);
  std::for_each(order, oEnd, [&](int val) { orderIndexMissing.flip(val); });
  bool orderInRange = std::all_of(order, oEnd, [&](int i) { return i >= 0 && i < nbDims; });

  if (shapeVolume != input.count || orderIndexMissing.any() || !orderInRange) {
    return false;
  }

  switch (input.type) {
    case nvinfer1::DataType::kFLOAT:
      // Fallthrough, kFLOAT is the same size as kINT32 so reshape is the same
    case nvinfer1::DataType::kINT32:
      return ReshapeWeightsInternal<uint32_t>(static_cast<const uint32_t*>(input.values), shape,
                                              order, static_cast<uint32_t*>(data), nbDims);
    case nvinfer1::DataType::kHALF:
      return ReshapeWeightsInternal<uint16_t>(static_cast<const uint16_t*>(input.values), shape,
                                              order, static_cast<uint16_t*>(data), nbDims);
    case nvinfer1::DataType::kINT8:
      return ReshapeWeightsInternal<uint8_t>(static_cast<const uint8_t*>(input.values), shape,
                                             order, static_cast<uint8_t*>(data), nbDims);
  }
  return false;
}

// All weights data in Forward are stored and handled in FwdWeights
// FwdWeights can be used as nvinfer1::Weights
class FwdWeights {
 public:
  FwdWeights() {}

  explicit FwdWeights(const std::vector<float>& data) {
    type_ = nvinfer1::DataType::kFLOAT;
    count_ = data.size();
    const char* data_ptr = reinterpret_cast<const char*>(data.data());
    data_.assign(data_ptr, data_ptr + data.size() * sizeof(float));
  }

  explicit FwdWeights(const std::vector<int>& data) {
    type_ = nvinfer1::DataType::kINT32;
    count_ = data.size();
    const char* data_ptr = reinterpret_cast<const char*>(data.data());
    data_.assign(data_ptr, data_ptr + data.size() * sizeof(int));
  }

  // return as nvinfer1::Weights
  operator nvinfer1::Weights() const { return {type_, data_.data(), count_}; }

  // Transpose weights to the correct order.
  bool Transpose(nvinfer1::Dims dims, const std::vector<int>& order) {
    auto weightsBytes =
        std::shared_ptr<uint8_t>{new uint8_t[data_.size()], [](const uint8_t* p) { delete[] p; }};

    if (!ReshapeWeights({type_, data_.data(), count_}, reinterpret_cast<int*>(dims.d), order.data(),
                        weightsBytes.get(), dims.nbDims)) {
      return false;
    }

    memcpy(data_.data(), weightsBytes.get(), data_.size());
    return true;
  }

  bool Empty() const { return count_ == 0; }

  const void* Data() const { return data_.data(); }

  void SetData(const char* data, int num_bytes) { data_.assign(data, data + num_bytes); }

  nvinfer1::DataType Type() const { return type_; }

  void SetType(const nvinfer1::DataType& type) { type_ = type; }

  int64_t Count() const { return count_; }

  void SetCount(int64_t count) { count_ = count; }

  const nvinfer1::Dims& Dims() const { return dims_; }

  void SetDims(const nvinfer1::Dims& dims) { dims_ = dims; }

  void CopyTo(float* dest, size_t ele_size) const {
    std::copy(reinterpret_cast<const float*>(data_.data()),
              (reinterpret_cast<const float*>(data_.data())) + ele_size, dest);
  }

 private:
  nvinfer1::DataType type_{nvinfer1::DataType::kFLOAT};
  int64_t count_{0};
  std::vector<char> data_;
  nvinfer1::Dims dims_{-1};
};

FWD_NAMESPACE_END
