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

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#include <numeric>
#include <vector>

#include "trt_engine/trt_network_crt/plugins/common/half_ext.cuh"

#define TRT_UNUSED (void)

using kv_float = cub::KeyValuePair<float, float>;
using kv_half = cub::KeyValuePair<half, half>;
using kv_half2 = cub::KeyValuePair<half2, half2>;

__device__ inline kv_float operator+(const kv_float& a, const kv_float& b) {
  return kv_float(a.key + b.key, a.value + b.value);
}

__device__ inline kv_half operator+(const kv_half& a, const kv_half& b) {
  const half2 a2 = __halves2half2(a.key, a.value);
  const half2 b2 = __halves2half2(b.key, b.value);
  const half2 res = __hadd2(a2, b2);
  return kv_half(res.x, res.y);
}

__device__ inline kv_half2 operator+(const kv_half2& a, const kv_half2& b) {
  return kv_half2(__hadd2(a.key, b.key), __hadd2(a.value, b.value));
}

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

FWD_TRT_NAMESPACE_BEGIN

#define DESER(d, m) m = readFromBuffer<decltype(m)>(d)

#define HDI inline __host__ __device__

// Helper function for serializing plugin
template <typename T>
inline void writeToBuffer(char*& buffer, const T& val) {
  *reinterpret_cast<T*>(buffer) = val;
  buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
inline T readFromBuffer(const char*& buffer) {
  T val = *reinterpret_cast<const T*>(buffer);
  buffer += sizeof(T);
  return val;
}

template <typename T>
__device__ inline T rsqrt(const T& x);

template <>
__device__ inline float rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline half rsqrt(const half& x) {
  return hrsqrt(x);
}

template <typename T>
__device__ inline T tanh(const T& x);

template <>
__device__ inline float tanh(const float& x) {
  return tanhf(x);
}

template <>
__device__ inline half tanh(const half& x) {
  const float tmp = tanhf(__half2float(x));
  return __float2half(tmp);
}

template <>
__device__ inline half2 tanh(const half2& x) {
  // at the moment, there is no half2 tanh builtin
  float2 tmp = (__half22float2(x));
  tmp.x = tanhf(tmp.x);
  tmp.y = tanhf(tmp.y);
  return __float22half2_rn(tmp);
}

template <typename T>
__device__ inline T exp(const T x);

template <>
__device__ inline float exp(const float x) {
  return expf(x);
}

template <>
__device__ inline half exp(const half x) {
  return hexp(x);
}

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2> {
  using type = uint16_t;
};
template <>
struct BytesToType<4> {
  using type = uint32_t;
};
template <>
struct BytesToType<8> {
  using type = uint64_t;
};
template <>
struct BytesToType<16> {
  using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data) {
  using T = typename BytesToType<Bytes>::type;

  const T* in = static_cast<const T*>(local);
  T* out = static_cast<T*>(data);
  *out = *in;
}

FWD_TRT_NAMESPACE_END
