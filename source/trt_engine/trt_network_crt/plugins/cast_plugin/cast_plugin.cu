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

#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

#include <cuda_fp16.h>

FWD_TRT_NAMESPACE_BEGIN
template <typename in_t, typename out_t>
__global__ void CastKernel(const in_t* input, out_t* out, size_t size) {
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    out[idx] = static_cast<out_t>(input[idx]);
  }
}

template <typename in_T, typename out_T>
void Cast(const in_T* input, out_T* output, size_t size) {
  const int blockDim = 1024;
  const int gridDim = static_cast<int>((size + blockDim - 1) / blockDim);

  CastKernel<in_T, out_T>
      <<<gridDim, blockDim>>>(static_cast<const in_T*>(input), static_cast<out_T*>(output), size);
}

template void Cast<half, float>(const half* input, float* output, size_t size);
template void Cast<int, float>(const int* input, float* output, size_t size);
template void Cast<int8_t, float>(const int8_t* input, float* output, size_t size);
template void Cast<bool, float>(const bool* input, float* output, size_t size);

FWD_TRT_NAMESPACE_END
