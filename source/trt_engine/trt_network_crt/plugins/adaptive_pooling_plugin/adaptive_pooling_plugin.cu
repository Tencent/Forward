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

#include "trt_engine/trt_network_crt/plugins/adaptive_pooling_plugin/adaptive_pooling_plugin.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "trt_engine/trt_network_crt/plugins/common/half_ext.cuh"
#include "trt_engine/trt_network_crt/plugins/common/trt_tensor_info.h"

FWD_TRT_NAMESPACE_BEGIN
__device__ inline int start_index(int a, int b, int c) { return (int)floorf((float)(a * c) / b); }

__device__ inline int end_index(int a, int b, int c) {
  return (int)ceilf((float)((a + 1) * c) / b);
}

// 4d tensor B x D x H x W

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2
 * and 3 4D input, 4D output, 4D argmax x and y
 */
template <typename T>
__global__ void adaptive_max_pool2d(const T *input, T *output, int isizeH, int isizeW, int osizeH,
                                    int osizeW, int64_t istrideD, int64_t istrideH,
                                    int64_t istrideW) {
  // iterators
  int oh, ow;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  int ostartH = blockDim.y * blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y * gridDim.y;
  // select input/output plane
  output = output + o_plane * osizeH * osizeW;
  input = input + i_plane * istrideD;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);

      int kW = iendW - istartW;

      // Compute the mean of the input image...
      const T *ptr_input = input + istartH * istrideH + istartW * istrideW;
      T *ptr_output = output + oh * osizeW + ow;
      T max = NumericLimits<T>::lowest();
      int ih, iw;
      for (ih = 0; ih < kH; ih++) {
        for (iw = 0; iw < kW; iw++) {
          T val = ptr_input[iw * istrideW];
          if ((val > max) || isnan(val)) {
            max = val;
          }
        }
        ptr_input += istrideH;  // next input line
      }
      // Update output and argmax
      *ptr_output = max;
    }
  }
}

// 5d tensor B x D x T x H x W

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2
 * and 3 4D input, 4D output, 4D argmax x and y
 */
template <typename T>
__global__ void adaptive_max_pool3d(const T *input, T *output, int isizeT, int isizeH, int isizeW,
                                    int osizeT, int osizeH, int osizeW, int64_t istrideD,
                                    int64_t istrideT, int64_t istrideH, int64_t istrideW,
                                    int64_t offsetZ) {
  // iterators on output pixels
  int ot, oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT;     // output frame/time
  int d = o_plane / osizeT;  // slice/feature

  // input frame/time ramge is fixed.
  int istartT = start_index(ot, osizeT, isizeT);
  int iendT = end_index(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  // input offset by slice/feature and earliest relevant frame/time
  const T *input_dt = input + d * istrideD + istartT * istrideT;
  // output offset by slice/feature and frame/time
  T *output_dt = output + o_plane * osizeH * osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      const T *ptr_input = input_dt + istartH * istrideH + istartW * istrideW;
      T *ptr_output = output_dt + oh * osizeW + ow;
      T max = NumericLimits<T>::lowest();

      int it, ih, iw;
      for (it = 0; it < kT; ++it) {
        for (ih = 0; ih < kH; ++ih) {
          for (iw = 0; iw < kW; ++iw) {
            T val = ptr_input[ih * istrideH + iw * istrideW];
            if ((val > max) || isnan(val)) {
              max = val;
            }
          }
        }
        ptr_input += istrideT;  // next input frame
      }
      // Update output
      *ptr_output = max;
    }
  }
}

// 4d tensor B x D x H x W
// All kernels view batch dim B and feature dim D as collapsed.

/*
 * Description:
 *    this function adaptively average pools an input 4D tensor along dimensions
 * 2 and 3 4D input, 4D output
 */
template <typename T>
__global__ void adaptive_average_pool2d(const T *input, T *output, int isizeH, int isizeW,
                                        int osizeH, int osizeW, int64_t istrideD, int64_t istrideH,
                                        int64_t istrideW) {
  // iterators on output pixels
  int oh, ow;

  // select input/output plane based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  output = output + o_plane * osizeH * osizeW;
  input = input + i_plane * istrideD;

  int ostartH = blockDim.y * blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y * gridDim.y;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling over corresponding input pixels
      const T *ptr_input = input + istartH * istrideH + istartW * istrideW;
      T *ptr_output = output + oh * osizeW + ow;
      T sum = 0;
      int ih, iw;
      for (ih = 0; ih < kH; ++ih) {
        for (iw = 0; iw < kW; ++iw) {
          T val = ptr_input[iw * istrideW];
          sum += val;
        }
        ptr_input += istrideH;  // next input line
      }
      // Update output
      *ptr_output = sum / kH / kW;
    }
  }
}

// 5d tensor B x D x T x H x W
// All kernels view batch dim B and dim D as collapsed.

/*
 * Description:
 *    this function adaptively average pools an input 5D tensor along dimensions
 * 2, 3, and 4 5D input, 5D output
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 */
template <typename T>
__global__ void adaptive_average_pool3d(const T *input, T *output, int isizeT, int isizeH,
                                        int isizeW, int osizeT, int osizeH, int osizeW,
                                        int64_t istrideD, int64_t istrideT, int64_t istrideH,
                                        int64_t istrideW, int64_t offsetZ) {
  // iterates on output pixels
  int ot, oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT;     // output frame/time
  int d = o_plane / osizeT;  // slice/feature

  // input frame/time range is fixed.
  int istartT = start_index(ot, osizeT, isizeT);
  int iendT = end_index(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  // input offset by slice/feature and earliest relevant frame/time
  const T *input_dt = input + d * istrideD + istartT * istrideT;
  // output offset by slice/feature and frame/time
  T *output_dt = output + o_plane * osizeH * osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      const T *ptr_input = input_dt + istartH * istrideH + istartW * istrideW;
      T *ptr_output = output_dt + oh * osizeW + ow;
      T sum = 0;

      int it, ih, iw;
      for (it = 0; it < kT; ++it) {
        for (ih = 0; ih < kH; ++ih) {
          for (iw = 0; iw < kW; ++iw) {
            T val = ptr_input[ih * istrideH + iw * istrideW];
            sum += val;
          }
        }
        ptr_input += istrideT;  // next input frame
      }
      // Update output
      *ptr_output = sum / kT / kH / kW;
    }
  }
}

// TODO: �� kernel ʵ�ֽ����Ż�

template <typename T>
void AdaptivePooling2DCuda(const TensorInfo<T> &input, TensorInfo<T> &output,
                           const std::vector<int> &output_size, PoolingOperation type,
                           cudaStream_t stream) {
  const int osizeH = output_size[0];
  const int osizeW = output_size[1];

  const int sizeB = input.Size(0);
  const int sizeD = input.Size(1);
  const int isizeH = input.Size(2);
  const int isizeW = input.Size(3);

  const int64_t istrideD = input.Stride(1);
  const int64_t istrideH = input.Stride(2);
  const int64_t istrideW = input.Stride(3);

  // cuda blocks & threads:
  int blocksH = std::max(16 / sizeD, 1);
  dim3 blocks(sizeB * sizeD, blocksH);
  dim3 threads(32, 8);

  if (type == PoolingOperation::MAX_POOLING) {
    adaptive_max_pool2d<T><<<blocks, threads, 0, stream>>>(input.DataPtr(), output.DataPtr(),
                                                           isizeH, isizeW, osizeH, osizeW, istrideD,
                                                           istrideH, istrideW);
  } else {
    adaptive_average_pool2d<T><<<blocks, threads, 0, stream>>>(input.DataPtr(), output.DataPtr(),
                                                               isizeH, isizeW, osizeH, osizeW,
                                                               istrideD, istrideH, istrideW);
  }
}

template <typename T>
void AdaptivePooling3DCuda(const TensorInfo<T> &input, TensorInfo<T> &output,
                           const std::vector<int> &output_size, PoolingOperation type,
                           cudaStream_t stream) {
  const int osizeT = output_size[0];
  const int osizeH = output_size[1];
  const int osizeW = output_size[2];

  const int sizeB = input.Size(0);
  const int sizeD = input.Size(1);
  const int isizeT = input.Size(2);
  const int isizeH = input.Size(3);
  const int isizeW = input.Size(4);

  const int64_t istrideD = input.Stride(1);
  const int64_t istrideT = input.Stride(2);
  const int64_t istrideH = input.Stride(3);
  const int64_t istrideW = input.Stride(4);

  int64_t totalZ = sizeB * sizeD * osizeT;
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = std::max(static_cast<int>(16L / totalZ), 1);

  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    if (type == PoolingOperation::MAX_POOLING) {
      adaptive_max_pool3d<T><<<blocks, threads, 0, stream>>>(
          input.DataPtr(), output.DataPtr(), isizeT, isizeH, isizeW, osizeT, osizeH, osizeW,
          istrideD, istrideT, istrideH, istrideW, offsetZ);
    } else {
      adaptive_average_pool3d<T><<<blocks, threads, 0, stream>>>(
          input.DataPtr(), output.DataPtr(), isizeT, isizeH, isizeW, osizeT, osizeH, osizeW,
          istrideD, istrideT, istrideH, istrideW, offsetZ);
    }
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

template void AdaptivePooling2DCuda<float>(const TensorInfo<float> &input,
                                           TensorInfo<float> &output,
                                           const std::vector<int> &output_size,
                                           PoolingOperation type, cudaStream_t stream);

template void AdaptivePooling2DCuda<half>(const TensorInfo<half> &input, TensorInfo<half> &output,
                                          const std::vector<int> &output_size,
                                          PoolingOperation type, cudaStream_t stream);

template void AdaptivePooling3DCuda<float>(const TensorInfo<float> &input,
                                           TensorInfo<float> &output,
                                           const std::vector<int> &output_size,
                                           PoolingOperation type, cudaStream_t stream);

template void AdaptivePooling3DCuda<half>(const TensorInfo<half> &input, TensorInfo<half> &output,
                                          const std::vector<int> &output_size,
                                          PoolingOperation type, cudaStream_t stream);

FWD_TRT_NAMESPACE_END
