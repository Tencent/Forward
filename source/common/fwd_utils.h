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

#include <numeric>
#include <string>
#include <vector>

#include "common/common_macros.h"

// Common utils for Forward

FWD_NAMESPACE_BEGIN

namespace FwdUtils {

//////////////////////////////////////////
//                                      //
//             数据类型操作相关          //
//          DataType Manipulation       //
//                                      //
//////////////////////////////////////////

// This is derived from:
// https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
static float Half2FloatFast(uint16_t value) {
  union F32 {
    unsigned int u;
    float f;
  };
  static const F32 magic = {(254 - 15) << 23};
  static const F32 was_infnan = {(127 + 16) << 23};
  F32 result;
  result.u = (value & 0x7fff) << 13;  // exponent/mantissa bits
  result.f *= magic.f;                // exponent adjust
  if (result.f >= was_infnan.f) {     // make sure Inf/NaN survive
    result.u |= 255 << 23;
  }
  result.u |= (value & 0x8000) << 16;  // sign bit
  return result.f;
}

// This is derived from numpy: npy_floatbits_to_halfbits
static uint16_t Float2Half(float f) {
  uint32_t* fi = reinterpret_cast<uint32_t*>(&f);
  uint32_t f_exp, f_sig;
  uint16_t h_sgn, h_exp, h_sig;

  h_sgn = (uint16_t)((*fi & 0x80000000u) >> 16);
  f_exp = (*fi & 0x7f800000u);

  // Exponent overflow/NaN converts to signed inf/NaN
  if (f_exp >= 0x47800000u) {
    if (f_exp == 0x7f800000u) {
      /* Inf or NaN */
      f_sig = (*fi & 0x007fffffu);
      if (f_sig != 0) {
        // NaN - propagate the flag in the significand...
        uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
        // ...but make sure it stays a NaN
        if (ret == 0x7c00u) {
          ret++;
        }
        return h_sgn + ret;
      } else {
        // signed inf
        return (uint16_t)(h_sgn + 0x7c00u);
      }
    } else {
      // overflow to signed inf
      return (uint16_t)(h_sgn + 0x7c00u);
    }
  }

  // Exponent underflow converts to a subnormal half or signed zero
  if (f_exp <= 0x38000000u) {
    // Signed zeros, subnormal floats, and floats with small
    // exponents all convert to signed zero half-floats.
    if (f_exp < 0x33000000u) {
      return h_sgn;
    }
    // Make the subnormal significand
    f_exp >>= 23;
    f_sig = (0x00800000u + (*fi & 0x007fffffu));

    // Usually the significand is shifted by 13. For subnormals an
    // additional shift needs to occur. This shift is one for the largest
    // exponent giving a subnormal `f_exp = 0x38000000 >> 23 = 112`, which
    // offsets the new first bit. At most the shift can be 1+10 bits.
    f_sig >>= (113 - f_exp);
    // Handle rounding by adding 1 to the bit beyond half precision

    f_sig += 0x00001000u;
    h_sig = (uint16_t)(f_sig >> 13);
    // If the rounding causes a bit to spill into h_exp, it will
    // increment h_exp from zero to one and h_sig will be zero.
    // This is the correct result.
    return (uint16_t)(h_sgn + h_sig);
  }

  // Regular case with no overflow or underflow
  h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
  // Handle rounding by adding 1 to the bit beyond half precision
  f_sig = (*fi & 0x007fffffu);
  f_sig += 0x00001000u;
  h_sig = (uint16_t)(f_sig >> 13);
  // If the rounding causes a bit to spill into h_exp, it will
  // increment h_exp by one and h_sig will be zero.  This is the
  // correct result.  h_exp may increment to 15, at greatest, in
  // which case the result overflows to a signed inf.
  return h_sgn + h_exp + h_sig;
}

//////////////////////////////////////////
//                                      //
//              字符串操作相关           //
//          String Manipulation         //
//                                      //
//////////////////////////////////////////

// return the replaced string of input.
// all src found in input will be replaced by dst.
static std::string ReplaceAll(const std::string& input, std::string src, std::string dst) {
  std::string::size_type pos(0);
  std::string output = input;
  while ((pos = output.find(src)) != std::string::npos) {
    output.replace(pos, src.length(), dst);
  }
  return output;
}

//////////////////////////////////////////
//                                      //
//              操作符重载相关           //
//          Operator Overloading        //
//                                      //
//////////////////////////////////////////

// return the pair-wise difference vector of v1 and v2.
template <typename T>
inline std::vector<T> operator-(const std::vector<T>& v1, const std::vector<T>& v2) {
  CHECK_EQ(v1.size(), v2.size());
  std::vector<T> res(v1.size());
  std::transform(v1.cbegin(), v1.cend(), v2.cbegin(), res.begin(),
                 [](const T& e1, const T& e2) { return e1 - e2; });
  return res;
}
}  // namespace FwdUtils

FWD_NAMESPACE_END
