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

#include "trt_engine/trt_network_crt/plugins/reflection_padding_plugin/reflection_padding_plugin.h"

#include <cuda_fp16.h>
#include <device_launch_parameters.h>

// #define ENABLE_REFLECTION_PADDING_2D_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

__device__ inline void get_index_mapping2d(int input_dim_x, int input_dim_y, int output_dim_x,
                                           int output_dim_y, int pad_l, int pad_t, int output_xy,
                                           int64_t* input_idx, int64_t* output_idx) {
  // 3D grid of 1D blocks
  const int64_t input_offset = (blockIdx.y + blockIdx.z * gridDim.y) * input_dim_x * input_dim_y;
  const int64_t output_offset = (blockIdx.y + blockIdx.z * gridDim.y) * output_dim_x * output_dim_y;

  const auto output_x = output_xy % output_dim_x;
  const auto output_y = output_xy / output_dim_x;

  const int i_start_x = ::max(0, -pad_l);
  const int i_start_y = ::max(0, -pad_t);
  const int o_start_x = ::max(0, pad_l);
  const int o_start_y = ::max(0, pad_t);

  auto input_x = ::abs(output_x - pad_l) - ::abs(output_x - (input_dim_x + pad_l - 1)) - output_x +
                 2 * pad_l + input_dim_x - 1 - o_start_x + i_start_x;

  auto input_y = ::abs(output_y - pad_t) - ::abs(output_y - (input_dim_y + pad_t - 1)) - output_y +
                 2 * pad_t + input_dim_y - 1 - o_start_y + i_start_y;

  *input_idx = input_offset + input_y * input_dim_x + input_x;
  *output_idx = output_offset + output_y * output_dim_x + output_x;
}

template <typename scalar_t>
__global__ void reflection_pad2d_out_kernel(const scalar_t* input, scalar_t* output,
                                            int input_dim_x, int input_dim_y, int pad_t, int pad_b,
                                            int pad_l, int pad_r) {
  const auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
  const auto output_dim_x = input_dim_x + pad_l + pad_r;
  const auto output_dim_y = input_dim_y + pad_t + pad_b;

  if (output_xy < output_dim_x * output_dim_y) {
    int64_t input_idx;
    int64_t output_idx;

    get_index_mapping2d(input_dim_x, input_dim_y, output_dim_x, output_dim_y, pad_l, pad_t,
                        output_xy, &input_idx, &output_idx);

    output[output_idx] = input[input_idx];
  }
}

template <typename T>
void ReflectionPad2D(const T* input, T* output, const nvinfer1::Dims& input_dims,
                     const std::vector<int>& padding_dims, const cudaStream_t& stream) {
  const int nbatch = input_dims.d[0];
  const int nplane = input_dims.d[1];
  const int input_h = input_dims.d[2];
  const int input_w = input_dims.d[3];

  const int pad_l = padding_dims[0];
  const int pad_r = padding_dims[1];
  const int pad_t = padding_dims[2];
  const int pad_b = padding_dims[3];

  ASSERT(pad_l < input_w && pad_r < input_w);
  ASSERT(pad_t < input_h && pad_b < input_h);

  const int output_h = input_h + pad_t + pad_b;
  const int output_w = input_w + pad_l + pad_r;

  const int output_plane_size = output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
  dim3 grid_size(static_cast<int>(std::ceil(output_plane_size / 256.0)), nplane, nbatch);

  reflection_pad2d_out_kernel<<<grid_size, block_size, 0, stream>>>(input, output, input_w, input_h,
                                                                    pad_t, pad_b, pad_l, pad_r);
}

template void ReflectionPad2D<float>(const float* input, float* output,
                                     const nvinfer1::Dims& input_dims,
                                     const std::vector<int>& padding_dims,
                                     const cudaStream_t& stream);

template void ReflectionPad2D<half>(const half* input, half* output,
                                    const nvinfer1::Dims& input_dims,
                                    const std::vector<int>& padding_dims,
                                    const cudaStream_t& stream);

FWD_TRT_NAMESPACE_END
