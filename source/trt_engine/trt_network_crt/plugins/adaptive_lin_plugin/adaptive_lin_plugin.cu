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

#include "trt_engine/trt_network_crt/plugins/adaptive_lin_plugin/adaptive_lin_plugin.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <cstdint>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/plugins/common/acc_types.h"
#include "trt_engine/trt_network_crt/plugins/common/reduce_common.cuh"

FWD_TRT_NAMESPACE_BEGIN

constexpr int kCUDANumThreads = 256;
constexpr int kCUDABlockReduceNumThreads = 512;
// constexpr int kColwiseReduceTileSize = 32;

template <typename T>
__global__ void CalcLINMoments(int C, int64_t HW, T eps, const T *X, acc_type<T> *mean,
                               acc_type<T> *rstd, acc_type<T> *reduce_sum1,
                               acc_type<T> *reduce_sum2) {
  using T_ACC = acc_type<T>;
  __shared__ T_ACC m_shared[WARP_SIZE];
  __shared__ T_ACC v_shared[WARP_SIZE];
  const int64_t i = blockIdx.x;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  for (int64_t j = threadIdx.x; j < HW; j += blockDim.x) {
    const int64_t index = i * HW + j;
    sum1 += static_cast<T_ACC>(X[index]);
    sum2 += static_cast<T_ACC>(X[index]) * static_cast<T_ACC>(X[index]);
  }
  sum1 = BlockReduceSum<T_ACC>(sum1, m_shared);
  sum2 = BlockReduceSum<T_ACC>(sum2, v_shared);
  if (threadIdx.x == 0) {
    const T_ACC scale = T_ACC(1) / static_cast<T_ACC>(HW);
    sum1 *= scale;
    sum2 *= scale;
    mean[i] = sum1;
    rstd[i] = rsqrt(max(sum2 - sum1 * sum1, T_ACC(0)) + static_cast<T_ACC>(eps));

    const int batch = blockIdx.x / C;
    atomicAdd(&reduce_sum1[batch], sum1);
    atomicAdd(&reduce_sum2[batch], sum2);
  }
}

template <typename T>
__global__ void LayerInstanceNorm(int C, int64_t HW, const T *X, const acc_type<T> *mean,
                                  const acc_type<T> *rstd, acc_type<T> *reduce_mean,
                                  acc_type<T> *reduce_rstd, T *in_out, T *ln_out, float eps) {
  using T_ACC = acc_type<T>;
  __shared__ T_ACC r_mean_v;
  __shared__ T_ACC r_rstd_v;
  if (threadIdx.x == 0) {
    const T_ACC scale = T_ACC(1) / static_cast<T_ACC>(C);
    const int batch = blockIdx.x / C;
    r_mean_v = reduce_mean[batch];
    r_rstd_v = reduce_rstd[batch];
    r_mean_v *= scale;
    r_rstd_v = max(r_rstd_v * scale - r_mean_v * r_mean_v, T_ACC(0));
    r_rstd_v = rsqrt(r_rstd_v + static_cast<T_ACC>(eps));
  }
  __syncthreads();

  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < HW; j += blockDim.x) {
    const int64_t index = i * HW + j;
    T_ACC x_v = static_cast<T_ACC>(X[index]);
    in_out[index] = (x_v - static_cast<T_ACC>(mean[i])) * static_cast<T_ACC>(rstd[i]);
    ln_out[index] = (x_v - static_cast<T_ACC>(r_mean_v)) * r_rstd_v;
  }

  // 这里不将结果写回，因为有线程块数据竞争风险
  // if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
  //   *reduce_mean = r_mean_v;
  //   *reduce_rstd = r_rstd_v;
  // }
}

// template <typename T>
// void AdaptiveLINCUDA(const T *X, int64_t N, int64_t C, int64_t HW, T eps,
//                      acc_type<T> *mean, acc_type<T> *rstd,
//                      acc_type<T> *reduce_mean, acc_type<T> *reduce_rstd,
//                      T *in_out, T *ln_out, cudaStream_t stream) {
//   const int64_t NC = N * C;
//   CalcLINMoments<T><<<NC, kCUDABlockReduceNumThreads, 0, stream>>>(
//       C, HW, eps, X, mean, rstd, reduce_mean, reduce_rstd);
//   LayerInstanceNorm<T><<<NC, kCUDANumThreads, 0, stream>>>(
//       C, HW, X, mean, rstd, reduce_mean, reduce_rstd, in_out, ln_out, eps);
// }

template <typename T>
void AdaptiveLINCUDA(const T *X, int64_t N, int64_t C, int64_t HW, T eps, float *mean, float *rstd,
                     float *reduce_mean, float *reduce_rstd, T *in_out, T *ln_out,
                     cudaStream_t stream) {
  const int64_t NC = N * C;
  CalcLINMoments<T><<<NC, kCUDABlockReduceNumThreads, 0, stream>>>(C, HW, eps, X, mean, rstd,
                                                                   reduce_mean, reduce_rstd);
  LayerInstanceNorm<T><<<NC, kCUDANumThreads, 0, stream>>>(C, HW, X, mean, rstd, reduce_mean,
                                                           reduce_rstd, in_out, ln_out, eps);
}

template void AdaptiveLINCUDA<float>(const float *X, int64_t N, int64_t C, int64_t HW, float eps,
                                     float *mean, float *rstd, float *reduce_mean,
                                     float *reduce_rstd, float *in_out, float *ln_out,
                                     cudaStream_t stream);

template void AdaptiveLINCUDA<half>(const half *X, int64_t N, int64_t C, int64_t HW, half eps,
                                    float *mean, float *rstd, float *reduce_mean,
                                    float *reduce_rstd, half *in_out, half *ln_out,
                                    cudaStream_t stream);

FWD_TRT_NAMESPACE_END
