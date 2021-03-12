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

#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"
#include "trt_engine/trt_network_crt/plugins/common/trt_tensor_info.h"
#include "trt_engine/trt_network_crt/plugins/embedding_bag_plugin/embedding_bag_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

__device__ static float atomicMax(float* address, float val) {
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

template <typename T>
inline __device__ static T atomicOp(T* address, T val, int op) {
  // op = 0: sum; 1: max; 2: mean; 3: square_sum
  switch (op) {
    case 0:
      return atomicAdd(address, val);
    case 1:
      return atomicMax(address, val);
    case 2:
      return atomicAdd(address, val);
    case 3:
      return atomicAdd(address, val * val);
    default:
      return atomicAdd(address, val);
  }
}

__global__ void EmbeddingBagCuda(const int* input, const int* offset, float* output, int input_size,
                                 const float* data, int count, int dim, int data_offset, int op) {
  extern __shared__ float block_result[];

  // set initial output
  int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
  while (thread_index < dim) {
    block_result[thread_index] = 0;
    thread_index += blockDim.x * blockDim.y;
  }

  __syncthreads();

  // set the start and end offset of each block
  int block_start = offset[blockIdx.x];
  int block_end;
  if (blockIdx.x == gridDim.x - 1) {
    block_end = input_size;
  } else {
    block_end = offset[blockIdx.x + 1];
  }

  // mark the index for this thread
  thread_index = block_start + threadIdx.x;

  // find and add the vectors
  while (thread_index < block_end) {
    int current_index = input[thread_index];
    if (current_index >= data_offset && current_index < count + data_offset) {
      const float* current_data = data + dim * (current_index - data_offset);
      int current_pos = threadIdx.y;
      while (current_pos < dim) {
        atomicOp(&block_result[current_pos], current_data[current_pos], op);
        current_pos += blockDim.y;
      }
    }

    thread_index += blockDim.x;
  }

  __syncthreads();

  // copy result to global memory
  int pos = threadIdx.y * blockDim.x + threadIdx.x;
  while (pos < dim) {
    if (op == 1) {
      output[dim * blockIdx.x + pos] = block_result[pos] / max(block_end - block_start, 1);
    } else {
      output[dim * blockIdx.x + pos] = block_result[pos];
    }
    pos += blockDim.x * blockDim.y;
  }
}

__global__ void EmbeddingBagCudaWithFixedOffset(const int* input, const int offset, float* output,
                                                int input_size, const float* data, int count,
                                                int dim, int data_offset, int op) {
  extern __shared__ float block_result[];

  // set initial output
  int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
  while (thread_index < dim) {
    block_result[thread_index] = 0;
    thread_index += blockDim.x * blockDim.y;
  }

  __syncthreads();

  // set the start and end offset of each block
  int block_start = offset * blockIdx.x;
  int block_end = block_start + offset;

  // mark the index for this thread
  thread_index = block_start + threadIdx.x;

  // find and add the vectors
  while (thread_index < block_end) {
    int current_index = input[thread_index];
    if (current_index >= data_offset && current_index < count + data_offset) {
      const float* current_data = data + dim * (current_index - data_offset);
      int current_pos = threadIdx.y;
      while (current_pos < dim) {
        atomicOp(&block_result[current_pos], current_data[current_pos], op);
        current_pos += blockDim.y;
      }
    }

    thread_index += blockDim.x;
  }

  __syncthreads();

  // copy result to global memory
  int pos = threadIdx.y * blockDim.x + threadIdx.x;
  while (pos < dim) {
    if (op == 1) {
      output[dim * blockIdx.x + pos] = block_result[pos] / max(block_end - block_start, 1);
    } else {
      output[dim * blockIdx.x + pos] = block_result[pos];
    }
    pos += blockDim.x * blockDim.y;
  }
}

__global__ void GatherKernel(const int* input, float* output, int input_size, const float* data,
                             int count, int dim, int data_offset) {
  // set initial output
  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index < input_size * dim) {
    const int input_id = input[thread_index / dim];
    const int pos = thread_index % dim;
    if (input_id < count + data_offset && input_id >= data_offset) {
      output[thread_index] = data[input_id * dim + pos];
    }
  }
}
void Gather(const int* input, float* output, int batch_size, int offset, const float* data,
            int count, int dim, int data_offset, const cudaStream_t& stream) {
  int blockDim = 32;
  int gridDim = (batch_size * offset * dim - 1) / 32 + 1;
  GatherKernel<<<gridDim, blockDim, 0, stream>>>(input, output, batch_size * offset, data, count,
                                                 dim, data_offset);
}

void EmbeddingBagWithFixedOffset(const int* input, float* output, int batch_size, int offset,
                                 const float* data, int count, int dim, int data_offset,
                                 int op_type, const cudaStream_t& stream) {
  dim3 thread_size(32, std::min(8, dim));
  EmbeddingBagCudaWithFixedOffset<<<batch_size, thread_size, sizeof(float) * dim, stream>>>(
      input, offset, output, batch_size * offset, data, count, dim, data_offset, op_type);
}

void EmbeddingBag(const int* input, const int* input_2, float* output, int input_size,
                  int block_size, const float* data, int count, int dim, int data_offset,
                  int op_type, const cudaStream_t& stream) {
  // 输入非二维时，考虑offset数组(对应torch.nn.embedding_bag)

  dim3 thread_size(32, std::min(8, dim));
  EmbeddingBagCuda<<<block_size, thread_size, sizeof(float) * dim, stream>>>(
      input, input_2, output, input_size, data, count, dim, data_offset, op_type);
}

FWD_TRT_NAMESPACE_END
