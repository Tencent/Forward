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

#include "trt_engine/trt_network_crt/plugins/constant_pad_plugin/constant_pad_plugin.h"

#include <cuda_fp16.h>

FWD_TRT_NAMESPACE_BEGIN

template <typename scalar_t>
__global__ void constant_pad2d_kernel(const scalar_t* input, scalar_t* output, scalar_t constant,
                                      int in_h, int in_w, int out_h, int out_w, int pad_t,
                                      int pad_b, int pad_l, int pad_r) {
  const auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;

  const auto input_offset = (blockIdx.y + blockIdx.z * gridDim.y) * in_h * in_w;
  const auto output_offset = (blockIdx.y + blockIdx.z * gridDim.y) * out_h * out_w;

  const int output_x = output_xy % out_w;
  const int output_y = output_xy / out_w;

  if (output_xy < out_h * out_w) {
    if (output_y < pad_t || output_y >= out_h - pad_b || output_x < pad_l ||
        output_x >= out_w - pad_r) {
      output[output_offset + output_xy] = constant;
    } else {
      const auto input_index = input_offset + (output_y - pad_t) * in_w + (output_x - pad_l);
      output[output_offset + output_xy] = input[input_index];
    }
  }
}

template <typename scalar_t>
__global__ void constant_pad3d_kernel(const scalar_t* input, scalar_t* output, scalar_t constant,
                                      int in_d, int in_h, int in_w, int out_d, int out_h, int out_w,
                                      int pad_t, int pad_b, int pad_l, int pad_r, int pad_f,
                                      int pad_k) {
  const auto output_xyz = threadIdx.x + blockIdx.x * blockDim.x;

  const auto input_offset = (blockIdx.y + blockIdx.z * gridDim.y) * in_h * in_w * in_d;
  const auto output_offset = (blockIdx.y + blockIdx.z * gridDim.y) * out_h * out_w * out_d;

  const int output_x = output_xyz % out_w;
  const int output_y = (output_xyz / out_w) % out_h;
  const int output_z = (output_xyz / out_w) / out_h;

  if (output_xyz < out_d * out_h * out_w) {
    if (output_z < pad_f || output_z >= out_d - pad_k || output_y < pad_t ||
        output_y >= out_h - pad_b || output_x < pad_l || output_x >= out_w - pad_r) {
      output[output_offset + output_xyz] = constant;
    } else {
      const auto input_index = input_offset + (output_z - pad_f) * in_h * in_w +
                               (output_y - pad_t) * in_w + (output_x - pad_l);
      output[output_offset + output_xyz] = input[input_index];
    }
  }
}

template <typename T>
void ConstantPad2D(const T* input, T* output, T constant, const nvinfer1::Dims& input_dims,
                   const std::vector<int>& padding_dims, const cudaStream_t& stream) {
  const int batch_size = input_dims.d[0];
  const int channel_size = input_dims.d[1];
  const int input_h = input_dims.d[2];
  const int input_w = input_dims.d[3];

  const int pad_l = padding_dims[0];
  const int pad_r = padding_dims[1];
  const int pad_t = padding_dims[2];
  const int pad_b = padding_dims[3];
  // TODO(Ao Li): 对负填充进行支持
  ASSERT(pad_l >= 0 && pad_r >= 0 && pad_t >= 0 && pad_b >= 0);

  const int output_h = input_h + pad_t + pad_b;
  const int output_w = input_w + pad_l + pad_r;

  const int output_plane_size = output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
  dim3 grid_size(static_cast<int>(std::ceil(output_plane_size / 256.0)), channel_size, batch_size);

  constant_pad2d_kernel<T><<<grid_size, block_size, 0, stream>>>(
      input, output, constant, input_h, input_w, output_h, output_w, pad_t, pad_b, pad_l, pad_r);
}

template <typename T>
void ConstantPad3D(const T* input, T* output, T constant, const nvinfer1::Dims& input_dims,
                   const std::vector<int>& padding_dims, const cudaStream_t& stream) {
  const int batch_size = input_dims.d[0];
  const int channel_size = input_dims.d[1];
  const int input_d = input_dims.d[2];
  const int input_h = input_dims.d[3];
  const int input_w = input_dims.d[4];

  const int pad_l = padding_dims[0];
  const int pad_r = padding_dims[1];
  const int pad_t = padding_dims[2];
  const int pad_b = padding_dims[3];
  const int pad_f = padding_dims[4];
  const int pad_k = padding_dims[5];

  // TODO(Ao Li): 支持负填充
  ASSERT(pad_l >= 0 && pad_r >= 0 && pad_t >= 0 && pad_b >= 0 && pad_f >= 0 && pad_k >= 0);

  const int output_d = input_d + pad_f + pad_k;
  const int output_h = input_h + pad_t + pad_b;
  const int output_w = input_w + pad_l + pad_r;

  const int output_plane_size = output_d * output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
  dim3 grid_size(static_cast<int>(std::ceil(output_plane_size / 256.0)), channel_size, batch_size);

  constant_pad3d_kernel<T><<<grid_size, block_size, 0, stream>>>(
      input, output, constant, input_d, input_h, input_w, output_d, output_h, output_w, pad_t,
      pad_b, pad_l, pad_r, pad_f, pad_k);
}

template void ConstantPad2D<float>(const float* input, float* output, float constant,
                                   const nvinfer1::Dims& input_dims,
                                   const std::vector<int>& padding_dims,
                                   const cudaStream_t& stream);

template void ConstantPad2D<half>(const half* input, half* output, half constant,
                                  const nvinfer1::Dims& input_dims,
                                  const std::vector<int>& padding_dims, const cudaStream_t& stream);

template void ConstantPad3D<float>(const float* input, float* output, float constant,
                                   const nvinfer1::Dims& input_dims,
                                   const std::vector<int>& padding_dims,
                                   const cudaStream_t& stream);

template void ConstantPad3D<half>(const half* input, half* output, half constant,
                                  const nvinfer1::Dims& input_dims,
                                  const std::vector<int>& padding_dims, const cudaStream_t& stream);

FWD_TRT_NAMESPACE_END
