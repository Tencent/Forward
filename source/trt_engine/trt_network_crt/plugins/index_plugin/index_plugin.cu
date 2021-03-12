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

#include "trt_engine/trt_network_crt/plugins/index_plugin/index_plugin.h"

#include <algorithm>

FWD_TRT_NAMESPACE_BEGIN

// 根据索引计算元素的位置
__device__ static void IndexToPos(size_t index, int* pos, const int* dims, int nb_dims) {
  for (int i = nb_dims - 1; i >= 0; i--) {
    pos[i] = index % dims[i];
    index /= dims[i];
  }
}

// 根据元素位置计算索引
__device__ static size_t PosToIndex(int* pos, const int* dims, int nb_dims) {
  size_t index = 0;
  for (int i = 0; i < nb_dims; i++) {
    index *= dims[i];
    index += pos[i];
  }
  return index;
}

// 根据新位置计算提取索引前的位置
__device__ static void NewPosToOldPos(int* old_pos, const int* new_pos, int nb_old_dims,
                                      int nb_new_dims, const int* index_data, const int* index_dim,
                                      int nb_index, bool put_front) {
  // find where the specified index is. If put_front, then it is at the front.
  int index_pos = 0;
  if (!put_front) {
    for (int i = 0; i < nb_new_dims; i++) {
      if (index_dim[i]) {
        index_pos = i;
        break;
      }
    }
  }
  int index = new_pos[index_pos];

  for (int i = 0; i < nb_old_dims; i++) {
    if (index_dim[i]) {
      old_pos[i] = index_data[index];
      index += nb_index;
    }
  }

  int old_pos_ptr = 0;
  for (int i = 0; i < nb_new_dims; i++) {
    if (i == index_pos) continue;
    while (index_dim[old_pos_ptr]) old_pos_ptr++;
    old_pos[old_pos_ptr] = new_pos[i];
  }
}

__global__ void IndexKernel(const float* input, float* output, int* index, int* input_dims,
                            int* output_dims, int* index_pos, int nb_input_dims, int nb_output_dims,
                            int nb_index, bool put_front) {
  size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t total_count = 1;
  for (int i = 0; i < nb_output_dims; i++) total_count *= output_dims[i];

  int output_pos[8];
  int input_pos[8];

  while (thread_idx < total_count) {
    IndexToPos(thread_idx, output_pos, output_dims, nb_output_dims);
    NewPosToOldPos(input_pos, output_pos, nb_input_dims, nb_output_dims, index, index_pos, nb_index,
                   put_front);
    size_t index = PosToIndex(input_pos, input_dims, nb_input_dims);
    output[thread_idx] = input[index];

    thread_idx += blockDim.x * gridDim.x;
  }
}

void IndexCuda(const float* input, float* output, int* index, int* input_dims, int* output_dims,
               int* index_pos, int nb_input_dims, int nb_output_dims, int nb_index, bool put_front,
               const cudaStream_t& stream) {
  int block_size = 65536;
  int thread_size = 128;
  IndexKernel<<<block_size, thread_size, 0, stream>>>(input, output, index, input_dims, output_dims,
                                                      index_pos, nb_input_dims, nb_output_dims,
                                                      nb_index, put_front);
}

FWD_TRT_NAMESPACE_END
