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

// device pointer and dimensions info for plugin

#include <NvInfer.h>
#include <device_launch_parameters.h>
#include <string.h>

#include <functional>
#include <numeric>

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief Tensor信息，用于在plugin处理数据
 */
template <typename T>
struct TensorInfo {
  TensorInfo() = delete;

  TensorInfo(T* device_ptr, const nvinfer1::Dims& dims) : data_ptr_(device_ptr) {
    AssignSizes(dims);
    CalcStrides();
  }

  TensorInfo(const T* device_ptr, const nvinfer1::Dims& dims)
      : data_ptr_(const_cast<T*>(device_ptr)) {
    AssignSizes(dims);
    CalcStrides();
  }

  __host__ __device__ T& At(int n, int c, int h, int w) {
    return data_ptr_[n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3]];
  }

  __host__ __device__ const T& At(int n, int c, int h, int w) const {
    return data_ptr_[n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3]];
  }

  __host__ __device__ const T* DataPtr() const { return data_ptr_; }

  __host__ __device__ T* DataPtr() { return data_ptr_; }

  __host__ __device__ int Dims() const { return dims_; }

  __host__ __device__ int64_t NumberElements() const { return numel_; }

  __host__ __device__ int Size(int index) const { return sizes_[index]; }

  __host__ __device__ int64_t Stride(int index) const { return strides_[index]; }

 private:
  T* data_ptr_;

  int dims_;

  int sizes_[nvinfer1::Dims::MAX_DIMS];

  int64_t strides_[nvinfer1::Dims::MAX_DIMS];

  int64_t numel_;

  void AssignSizes(const nvinfer1::Dims& dims) {
    dims_ = dims.nbDims;
    memcpy(sizes_, dims.d, dims_ * sizeof(int));
    numel_ = dims_ == 0
                 ? 0
                 : std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
  }

  void CalcStrides() {
    int64_t stride = 1;
    for (int i = dims_ - 1; i >= 0; --i) {
      strides_[i] = stride;
      stride *= sizes_[i];
    }
  }
};

FWD_TRT_NAMESPACE_END
