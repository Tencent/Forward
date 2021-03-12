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

#include "trt_engine/trt_network_crt/plugins/split_plugin/split_plugin.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "trt_engine/trt_network_crt/plugins/common/half_ext.cuh"

FWD_TRT_NAMESPACE_BEGIN

template <typename scalar_t>
__global__ void split_kernel(TensorInfo<scalar_t> input, scalar_t* const* outputs, int* split_size,
                             int* output_pos, int* output_off, int dim) {
  const int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;

  const int64_t total = input.NumberElements();
  if (thread_idx < total) {
    const int64_t stride = input.Stride(dim);
    const int input_size = input.Size(dim);
    const int index = thread_idx / stride % input_size;
    const int output_index = output_pos[index];
    const int output_size = split_size[output_index];

    // calc output offset
    int offset = thread_idx % stride + output_off[index] * stride;

    int rest = 0;
    if (dim > 0) {
      rest = thread_idx - thread_idx % (input.Stride(dim - 1));
    }
    offset += rest / input_size * output_size;
    outputs[output_index][offset] = input.DataPtr()[thread_idx];
  }
}

template <typename T>
void SplitCuda(const TensorInfo<T>& input, T* const* outputs, int* split_size, int* output_pos,
               int* output_off, int dim, cudaStream_t stream) {
  const int blockDim = 1024;
  const int gridDim = (input.NumberElements() + blockDim - 1) / blockDim;
  split_kernel<T>
      <<<gridDim, blockDim, 0, stream>>>(input, outputs, split_size, output_pos, output_off, dim);
}

template void SplitCuda<float>(const TensorInfo<float>& input, float* const* outputs,
                               int* split_size, int* output_pos, int* output_off, int dim,
                               cudaStream_t stream);

template void SplitCuda<half>(const TensorInfo<half>& input, half* const* outputs, int* split_size,
                              int* output_pos, int* output_off, int dim, cudaStream_t stream);

FWD_TRT_NAMESPACE_END
