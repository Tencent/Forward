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

#include <NvInfer.h>

#include <cassert>
#include <cstring>
#include <vector>

#include <cub/cub.cuh>

#include "trt_engine/trt_network_crt/plugins/common/plugin_util.h"
#include "trt_engine/trt_network_crt/plugins/layer_norm_plugin/layer_norm_plugin.h"

using namespace nvinfer1;

FWD_TRT_NAMESPACE_BEGIN

template <typename T, typename R, typename P, int TPB>
__device__ inline void layerNorm(const kvp<R>& threadData, const int ld, const int offset,
                                 const P* beta, const P* gamma, T* output) {
  // Assuming threadData is already divided by ld

  using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ R mu;      // mean
  __shared__ R rsigma;  // 1 / std.dev.

  const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const R val = output[idx];
    const R g(gamma[i]);
    const R b(beta[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, typename P, int TPB>
__device__ inline void layerNormSmall(const T val, const kvp<T>& threadData, const int ld,
                                      const int idx, const P* beta, const P* gamma, T* output) {
  // Assuming threadData is already divided by ld
  // Small settings: the block covers the leading dimension TPB >= ld. The input
  // value is available in a register

  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    const T g(gamma[threadIdx.x]);
    const T b(beta[threadIdx.x]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <int TPB, int VPT, bool hasBias>
__global__ void lnDQQ(const int ld, const int8_t* input, int8_t* output, const __half* beta,
                      const __half* gamma, const __half* bias, const float dqScaleIn,
                      const float qScale) {
  const int idx = ld * blockIdx.x + threadIdx.x * VPT;
  // 4 * 1024 * 4 * 2 Bytes = 16KB per block
  int8_t in_local[VPT];

  __half in_local_dq[VPT];  // dequantized input + bias
  __half bias_local[VPT];   // bias and beta
  __half gamma_local[VPT];
  copy<sizeof(int8_t) * VPT>(&input[idx], in_local);
  copy<sizeof(__half) * VPT>(&bias[threadIdx.x * VPT], bias_local);
  __half2 loc = __floats2half2_rn(0.f, 0.f);  // accumulator

  const __half rld = __half(1) / __half(ld);
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    // DQ input
    const float tmp_in = in_local[it];
    in_local_dq[it] = dqScaleIn * tmp_in;

    if (hasBias) in_local_dq[it] += bias_local[it];
    const __half tmp = rld * in_local_dq[it];
    const __half2 tmp2 = __halves2half2(tmp, tmp * in_local_dq[it]);
    loc = loc + tmp2;
  }
  // load parameters
  copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], bias_local);
  copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], gamma_local);

  using BlockReduce = cub::BlockReduce<__half2, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ __half mu;      // mean
  __shared__ __half rsigma;  // 1 / std.dev.

  const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());

  if (threadIdx.x == 0) {
    mu = __low2half(sum2);
    rsigma = rsqrt(__high2half(sum2) - mu * mu);
  }
  __syncthreads();
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    // apply layernorm
    const float tmp = gamma_local[it] * (in_local_dq[it] - mu) * rsigma + bias_local[it];
    // Quantize
    int tmpq = __float2int_rn(qScale * tmp);
    tmpq = max(-127, tmpq);
    tmpq = min(127, tmpq);
    in_local[it] = tmpq;
  }

  copy<sizeof(int8_t) * VPT>(in_local, &output[idx]);
}

template <typename T, int TPB, int VPT, bool hasBias>
__global__ void ln_vec(const int ld, const T* input, T* output, const T* beta, const T* gamma,
                       const T* bias) {
  const int idx = ld * blockIdx.x + threadIdx.x * VPT;
  // 4 * 1024 * 4 * 2 Bytes = 16KB per block
  T in_local[VPT];
  T bias_local[VPT];
  T gamma_local[VPT];
  copy<sizeof(T) * VPT>(&input[idx], in_local);
  copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], bias_local);
  T local = 0.f;
  T local2 = 0.f;

  const T rld = T(1) / T(ld);
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    if (hasBias) in_local[it] += bias_local[it];
    const T tmp = rld * in_local[it];
    local += tmp;
    local2 += tmp * in_local[it];
  }

  copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], bias_local);
  copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], gamma_local);

  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<T>(local, local2), cub::Sum());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();
  ///*
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    in_local[it] = gamma_local[it] * (in_local[it] - mu) * rsigma + bias_local[it];
  }
  /* */

  copy<sizeof(T) * VPT>(in_local, &output[idx]);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void LayerNormKernelSmall(const int ld, const T* input, const T* beta, const T* gamma,
                                     T* output, const T* bias) {
  const T rld = T(1) / T(ld);
  const int offset = blockIdx.x * ld;

  cub::Sum pairSum;
  // reduce x and x^2
  kvp<T> threadData(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  if (threadIdx.x < ld) {
    val = input[idx];
    if (hasBias) {
      val += bias[threadIdx.x];
    }

    const T rldval = rld * val;
    threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
  }

  layerNormSmall<T, T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void LayerNormKernel(const int ld, const T* input, const T* beta, const T* gamma,
                                T* output, const T* bias) {
  const T rld = T(1) / T(ld);
  const int offset = blockIdx.x * ld;

  cub::Sum pairSum;
  // reduce x and x^2
  kvp<T> threadData(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    T val = T(input[idx]);

    if (hasBias) {
      val += T(bias[i]);
    }
    const T rldval = rld * val;
    threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    output[idx] = val;
  }

  layerNorm<T, T, T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <bool hasBias>
int computeLayerNormDQQ(cudaStream_t stream, const int ld, const int n, const int8_t* input,
                        const __half* beta, const __half* gamma, int8_t* output, const __half* bias,
                        const float dqScaleIn, const float qScale) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);

  const int gridSize = n / ld;
  // we're limited by the size of the parameters, i.e. 8-wide instead of 16
  constexpr int VPT = 16 / sizeof(__half);
  if (ld == 768) {
    constexpr int TPB = 768 / VPT;
    lnDQQ<TPB, VPT, hasBias>
        <<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias, dqScaleIn, qScale);
  } else if (ld == 1024) {
    constexpr int TPB = 1024 / VPT;
    lnDQQ<TPB, VPT, hasBias>
        <<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias, dqScaleIn, qScale);
  } else {
    // TODO need to implement this
    LOG(ERROR) << "SkipLayerNormDQQ - FATAL: unsupported hidden layer size: " << ld << std::endl;
    exit(0);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

template <typename T, bool hasBias>
int computeLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* beta,
                     const T* gamma, T* output, const T* bias) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int gridSize = n / ld;
  constexpr int VPT = 16 / sizeof(T);
  if (ld <= 32) {
    constexpr int blockSize = 32;
    LayerNormKernelSmall<T, blockSize, hasBias>
        <<<gridSize, blockSize, 0, stream>>>(ld, input, beta, gamma, output, bias);
  } else if (ld == 768) {
    constexpr int TPB = 768 / VPT;
    ln_vec<T, TPB, VPT, hasBias>
        <<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias);
  } else if (ld == 1024) {
    constexpr int TPB = 1024 / VPT;
    ln_vec<T, TPB, VPT, hasBias>
        <<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias);
  } else {
    constexpr int blockSize = 256;
    LayerNormKernel<T, blockSize, hasBias>
        <<<gridSize, blockSize, 0, stream>>>(ld, input, beta, gamma, output, bias);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

template int computeLayerNormDQQ<true>(cudaStream_t stream, const int ld, const int n,
                                       const int8_t* input, const __half* beta, const __half* gamma,
                                       int8_t* output, const __half* bias, const float dqScaleIn,
                                       const float qScale);
template int computeLayerNormDQQ<false>(cudaStream_t stream, const int ld, const int n,
                                        const int8_t* input, const __half* beta,
                                        const __half* gamma, int8_t* output, const __half* bias,
                                        const float dqScaleIn, const float qScale);

template int computeLayerNorm<float, true>(cudaStream_t, const int, const int, const float*,
                                           const float*, const float*, float*, const float*);
template int computeLayerNorm<float, false>(cudaStream_t, const int, const int, const float*,
                                            const float*, const float*, float*, const float*);
template int computeLayerNorm<half, true>(cudaStream_t, const int, const int, const half*,
                                          const half*, const half*, half*, const half*);
template int computeLayerNorm<half, false>(cudaStream_t, const int, const int, const half*,
                                           const half*, const half*, half*, const half*);

FWD_TRT_NAMESPACE_END
