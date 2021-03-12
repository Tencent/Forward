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

#include "trt_engine/trt_network_crt/plugins/upsampler_plugin/upsample_bilinear_2d.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "trt_engine/trt_network_crt/plugins/common/acc_types.h"
#include "trt_engine/trt_network_crt/plugins/common/half_ext.cuh"

FWD_TRT_NAMESPACE_BEGIN

#define LAUNCH_BOUNDS_0 \
  __launch_bounds__(256, 4)  // default launch bounds that should give good occupancy
                             // and versatility across all architectures.
#define LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((max_threads_per_block))
#define LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__((max_threads_per_block), (min_blocks_per_sm))

__device__ __forceinline__ size_t idx(const size_t nc, const size_t height, const size_t width,
                                      const size_t y, const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t compute_scales_value(float scale, int64_t src_size,
                                                                 int64_t dst_size) {
  return (scale != 1.0f) ? (accscalar_t)(1.0 / scale) : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t area_pixel_compute_scale(int input_size,
                                                                     int output_size,
                                                                     bool align_corners,
                                                                     const float scale) {
  if (output_size > 1) {
    return align_corners ? (accscalar_t)(input_size - 1) / (output_size - 1)
                         : compute_scales_value<accscalar_t>(scale, input_size, output_size);
  } else {
    return static_cast<accscalar_t>(0);
  }
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t area_pixel_compute_source_index(accscalar_t scale,
                                                                              int dst_index,
                                                                              bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  }
  accscalar_t src_idx =
      scale * (dst_index + static_cast<accscalar_t>(0.5)) - static_cast<accscalar_t>(0.5);
  return (src_idx < static_cast<accscalar_t>(0)) ? static_cast<accscalar_t>(0) : src_idx;
}

template <typename scalar_t, typename accscalar_t>

LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(const int n, const accscalar_t rheight,
                                              const accscalar_t rwidth, const bool align_corners,
                                              const TensorInfo<scalar_t> idata,
                                              TensorInfo<scalar_t> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.Size(0);
  const int channels = idata.Size(1);
  const int height1 = idata.Size(2);
  const int width1 = idata.Size(3);
  const int height2 = odata.Size(2);
  const int width2 = odata.Size(3);

  if (index < n) {
    const int w2 = index % width2;  // 0:width2-1
    const int h2 = index / width2;  // 0:height2-1
    // special case: just copy, unreachable
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata.At(n, c, h1, w1);
          odata.At(n, c, h2, w2) = val;
        }
      }
      return;
    }
    //
    const accscalar_t h1r =
        area_pixel_compute_source_index<accscalar_t>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = h0lambda * (w0lambda * idata.At(n, c, h1, w1) +
                                            w1lambda * idata.At(n, c, h1, w1 + w1p)) +
                                h1lambda * (w0lambda * idata.At(n, c, h1 + h1p, w1) +
                                            w1lambda * idata.At(n, c, h1 + h1p, w1 + w1p));
        odata.At(n, c, h2, w2) = static_cast<scalar_t>(val);
      }
    }
  }
}

template <typename T>
void UpSampleBilinear2DCuda(const TensorInfo<T>& input, TensorInfo<T>& output, int output_height,
                            int output_width, bool align_corners, float scale_h, float scale_w,
                            cudaStream_t stream) {
  const int input_height = input.Size(2);
  const int input_width = input.Size(3);

  const int num_kernels = output_height * output_width;
  const int num_threads = 1024;

  using accscalar_t = typename AccumulateType<T>::type;

  const accscalar_t rheight =
      area_pixel_compute_scale<accscalar_t>(input_height, output_height, align_corners, scale_h);
  const accscalar_t rwidth =
      area_pixel_compute_scale<accscalar_t>(input_width, output_width, align_corners, scale_w);

  upsample_bilinear2d_out_frame<T, accscalar_t>
      <<<(num_kernels + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
          num_kernels, rheight, rwidth, align_corners, input, output);
}

template void UpSampleBilinear2DCuda<float>(const TensorInfo<float>& input,
                                            TensorInfo<float>& output, int output_height,
                                            int output_width, bool align_corners, float scale_h,
                                            float scale_w, cudaStream_t stream);

template void UpSampleBilinear2DCuda<__half>(const TensorInfo<__half>& input,
                                             TensorInfo<__half>& output, int output_height,
                                             int output_width, bool align_corners, float scale_h,
                                             float scale_w, cudaStream_t stream);

FWD_TRT_NAMESPACE_END
