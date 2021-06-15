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
#include <cuda.h>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/plugins/common/acc_types.h"

FWD_TRT_NAMESPACE_BEGIN

__inline__ __device__ float pow(float X, float Y) { return (Y == 1.0f) ? X : powf(X, Y); }

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = WARP_SIZE,
                                            unsigned int mask = 0xffffffff) {
  return __shfl_down_sync(mask, value, delta, width);
}

template <typename T, int SIZE = WARP_SIZE>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  val = WarpReduceSum(val);
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T, int N>
__global__ void ReduceSumByFixWarpSize(const T* X, T* result, float power) {
  using T_ACC = acc_type<T>;

  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  const int doc_idx = threadIdx.x / N;
  const int doc_offset = threadIdx.x % N;
  const int num_doc = blockDim.x / N;

  T_ACC sum = pow(static_cast<T_ACC>(X[thread_idx]), power);

  // 仅支持 N = 32, 16, 8, 4, 2
  sum = WarpReduceSum<T_ACC, N>(sum);

  if (doc_offset == 0) {
    result[blockIdx.x * num_doc + doc_idx] = pow(sum, 1 / power);
  }
}

template <typename T, int LEN>
__global__ void ReduceSumUnroll(int64_t N, const T* X, float power, T* result) {
  using T_ACC = acc_type<T>;
  __shared__ T_ACC m_shared[WARP_SIZE];
  const int64_t i = blockIdx.x;
  T_ACC sum = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum += pow(static_cast<T_ACC>(X[index]), power);
  }

  sum = BlockReduceSum<T_ACC>(sum, m_shared);
  if (threadIdx.x == 0) {
    result[i] = pow(sum, 1 / power);
  }
}

template <typename T>
__global__ void ReduceSum(int64_t N, const T* X, float power, T* result) {
  using T_ACC = acc_type<T>;
  __shared__ T_ACC m_shared[WARP_SIZE];
  const int64_t i = blockIdx.x;
  T_ACC sum = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum += pow(static_cast<T_ACC>(X[index]), power);
  }

  sum = BlockReduceSum<T_ACC>(sum, m_shared);
  if (threadIdx.x == 0) {
    result[i] = pow(sum, 1 / power);
  }
}

template <typename T>
__global__ void TailReduceSum(int64_t PreDims, int64_t ReduceDims, int64_t PostDims, const T* X,
                              float power, T* result) {
  using T_ACC = acc_type<T>;
  __shared__ T_ACC m_shared[WARP_SIZE];
  for (int64_t i = blockIdx.x; i < PreDims; i += gridDim.x) {
    for (int64_t k = blockIdx.y; k < PostDims; k += gridDim.y) {
      T_ACC sum = 0;
      int64_t index = (i * ReduceDims + threadIdx.x) * PostDims + k;
      for (int64_t j = threadIdx.x; j < ReduceDims; j += blockDim.x) {
        sum += pow(static_cast<T_ACC>(X[index]), power);
        index += PostDims * blockDim.x;
      }
      sum = BlockReduceSum<T_ACC>(sum, m_shared);
      if (threadIdx.x == 0) {
        result[i * PostDims + k] = pow(sum, 1 / power);
      }
    }
  }
}

template <typename T>
__global__ void NonTailReduceSum(int64_t PreDims, int64_t ReduceDims, int64_t PostDims, const T* X,
                                 float power, T* result) {
  using T_ACC = acc_type<T>;
  constexpr int vt = 4;

  int pre_id;
  int post_id;
  T value_list[vt];

  for (int thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < PreDims * PostDims;
       thread_idx += blockDim.x * gridDim.x) {
    pre_id = thread_idx / PostDims;
    post_id = thread_idx % PostDims;

    int64_t data_index = (pre_id * ReduceDims) * PostDims + post_id;
    T_ACC sum = 0;
    int64_t j = 0;

    /*
     for (; j < ReduceDims; j++)
        {
            //sum += static_cast<T_ACC>(X[index]);
            sum += powf(static_cast<T_ACC>(X[index]), power);
            index += PostDims;
        }
     */

    for (; j < ReduceDims - vt + 1; j += vt) {
#pragma unroll
      for (int t = 0; t < vt; t++) {
        value_list[t] = X[data_index + t * PostDims];
      }

#pragma unroll
      for (int t = 0; t < vt; t++) {
        sum += pow(static_cast<T_ACC>(value_list[t]), power);
      }
      data_index += vt * PostDims;
    }

#pragma unroll
    for (int t = 0; t < vt; t++) {
      if (j + t >= ReduceDims) {
        break;
      }
      sum += pow(static_cast<T_ACC>(X[data_index]), power);
      data_index += PostDims;
    }

    result[pre_id * PostDims + post_id] = pow(sum, 1 / power);
  }
}

FWD_TRT_NAMESPACE_END
