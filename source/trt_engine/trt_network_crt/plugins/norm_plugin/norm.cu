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

#include "trt_engine/trt_network_crt/plugins/norm_plugin/norm_plugin.h"

#include <algorithm>

#include "trt_engine/trt_network_crt/plugins/common/acc_types.h"

FWD_TRT_NAMESPACE_BEGIN

__inline__ __device__ float pow(float X, float Y) { return (Y == 1.0f) ? abs(X) : powf(X, Y); }

template <typename T>
__global__ void NormKernel(int64_t PreDims, int64_t ReduceDims, int64_t PostDims, const T* X,
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

    data_index = (pre_id * ReduceDims) * PostDims + post_id;
    auto norm = pow(sum, 1 / power);
    for (j = 0; j < ReduceDims; j++) {
      result[data_index] = static_cast<T_ACC>(X[data_index]) / norm;
      data_index += PostDims;
    }

    // result[pre_id * PostDims + post_id] = pow(sum, 1/power);
  }
}

template <typename T>
void NormCuda(const T* X, int64_t M, int64_t N, int64_t K, T* output, float power,
              cudaStream_t stream) {
  const int blockDim = K > 1024 ? 1024 : static_cast<int>(1024 / K) * K;
  const int gridDim = std::min((M * K + blockDim - 1) / blockDim, static_cast<int64_t>(65535));
  NormKernel<T><<<gridDim, blockDim, 0, stream>>>(M, N, K, X, power, output);
}

template void NormCuda<float>(const float* X, int64_t M, int64_t N, int64_t K, float* output,
                              float power, cudaStream_t stream);

template void NormCuda<half>(const half* X, int64_t M, int64_t N, int64_t K, half* output,
                             float power, cudaStream_t stream);

FWD_TRT_NAMESPACE_END
