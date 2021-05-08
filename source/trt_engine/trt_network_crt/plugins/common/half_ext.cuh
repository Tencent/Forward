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

#pragma once

#include <common/common_macros.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

FWD_TRT_NAMESPACE_BEGIN

__device__ __forceinline__ float fabs(float a) { return ::fabsf(a); }

__device__ __forceinline__ half fabs(half a) { return __hle(a, 0.0f) ? __hneg(a) : a; }

__device__ __forceinline__ float floor(float a) { return ::floorf(a); }

__device__ __forceinline__ half floor(half a) { return hfloor(a); }

__device__ __forceinline__ float fmod(float a, float b) { return ::fmodf(a, b); }

__device__ __forceinline__ float fmod(half a, half b) {
  return ::fmod(__half2float(a), __half2float(b));
}

__device__ __forceinline__ bool isnan(half a) { return __hisnan(a); }

__device__ __forceinline__ float max(float a, float b) { return ::max(a, b); }

__device__ __forceinline__ float min(float a, float b) { return ::min(a, b); }

__device__ __forceinline__ half max(half a, half b) { return __hgt(a, b) ? a : b; }

__device__ __forceinline__ half min(half a, half b) { return __hlt(a, b) ? a : b; }

__device__ __forceinline__ half operator/(half a, int b) {
  return __float2half(__half2float(a) / b);
}

__device__ __forceinline__ half operator-(int a, half b) {
  return __float2half(a - __half2float(b));
}

__device__ __forceinline__ half operator-(half a, int b) {
  return __float2half(__half2float(a) - b);
}

__device__ __forceinline__ float operator*(float a, half b) { return a * __half2float(b); }

__device__ __forceinline__ float round(float a) { return round(a); }

__device__ __forceinline__ float round(const half &a) { return round(__half2float(a)); }

#include <cfloat>

template <typename T>
class NumericLimits {};

template <>
class NumericLimits<float> {
 public:
  __device__ __forceinline__ static float max() noexcept { return FLT_MAX; }

  __device__ __forceinline__ static float lowest() noexcept { return -FLT_MAX; }
};

template <>
class NumericLimits<half> {
 public:
  __device__ __forceinline__ static half max() noexcept { return __half_raw{0x7BFF}; }

  __device__ __forceinline__ static half lowest() noexcept { return __half_raw{0xFBFF}; }
};

FWD_TRT_NAMESPACE_END
