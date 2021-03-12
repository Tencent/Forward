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

#include "trt_engine/trt_network_crt/plugins/reduce_plugin/reduce_plugin.h"

#include <algorithm>

#include "trt_engine/trt_network_crt/plugins/common/reduce_common.cuh"

FWD_TRT_NAMESPACE_BEGIN

template <typename T>
void ReduceCuda(const T* X, int64_t M, int64_t N, int64_t K, T* output, float power,
                cudaStream_t stream) {
#ifdef SHUFFLE_REDUCE
  ReduceSum<T><<<M, std::min(N, static_cast<int64_t>(512l)), 0, stream>>>(N, X, output);
#else
  if (K == 0) {
    const int blockDim = 256;
    const int gridDim = std::max(M * N / blockDim, static_cast<int64_t>(1));
    switch (N) {
      case 32:
        ReduceSumByFixWarpSize<T, 32><<<gridDim, blockDim, 0, stream>>>(X, output, power);
        break;
      case 16:
        ReduceSumByFixWarpSize<T, 16><<<gridDim, blockDim, 0, stream>>>(X, output, power);
        break;
      case 8:
        ReduceSumByFixWarpSize<T, 8><<<gridDim, blockDim, 0, stream>>>(X, output, power);
        break;
      case 4:
        ReduceSumByFixWarpSize<T, 4><<<gridDim, blockDim, 0, stream>>>(X, output, power);
        break;
      case 2:
        ReduceSumByFixWarpSize<T, 2><<<gridDim, blockDim, 0, stream>>>(X, output, power);
        break;
      default:
        TailReduceSum<T>
            <<<std::min(M, static_cast<int64_t>(65535l)), std::min(N, static_cast<int64_t>(512l)),
               0, stream>>>(M, N, 1, X, power, output);
        break;
    }
  } else {
    // NonTailReduceSum<T>
    //     <<<std::min(M, static_cast<int64_t>(65535l)),
    //        std::min(K, static_cast<int64_t>(1024l)), 0, stream>>>(
    //         M, N, K, X, power, output);
    const int blockDim = K > 1024 ? 1024 : static_cast<int>(1024 / K) * K;
    const int gridDim = std::min((M * K + blockDim - 1) / blockDim, static_cast<int64_t>(65535));
    NonTailReduceSum<T><<<gridDim, blockDim, 0, stream>>>(M, N, K, X, power, output);
  }
#endif
}

template void ReduceCuda<float>(const float* X, int64_t M, int64_t N, int64_t K, float* output,
                                float power, cudaStream_t stream);

template void ReduceCuda<half>(const half* X, int64_t M, int64_t N, int64_t K, half* output,
                               float power, cudaStream_t stream);

FWD_TRT_NAMESPACE_END
