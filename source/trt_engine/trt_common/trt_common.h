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

#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/fwd_common.h"

#undef max
#undef min

FWD_NAMESPACE_BEGIN

constexpr static int WARP_SIZE = 32;

namespace TrtCommon {
inline unsigned int GetElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
}

inline fwd::DataType FwdDataType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kINT32:
      return fwd::DataType::INT32;
    case nvinfer1::DataType::kFLOAT:
      return fwd::DataType::FLOAT;
    case nvinfer1::DataType::kHALF:
      return fwd::DataType::HALF;
    case nvinfer1::DataType::kINT8:
      return fwd::DataType::INT8;
    default:
      return fwd::DataType::INVALID;
  }
}

inline std::string StringOf(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
#if NV_TENSORRT_MAJOR >= 7
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
#endif // NV_TENSORRT_MAJOR >= 7
    default:
      return "INVALID";
  }
}

inline nvinfer1::DataType GetDataType(bool use_fp16, bool use_int8, bool is_calib_mode) {
  nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
  if (use_fp16) dtype = nvinfer1::DataType::kHALF;
  if (use_int8 && !is_calib_mode) dtype = nvinfer1::DataType::kINT8;
  return dtype;
}
//////////////////////////////////////////
//                                      //
//            Set TRT Network           //
//                                      //
//////////////////////////////////////////

static void SetTensorName(nvinfer1::ITensor* tensor, const std::string& prefix,
                          const std::string& name) {
  tensor->setName((prefix + name).c_str());
}

static void SetOutputName(nvinfer1::ILayer* layer, const std::string& prefix,
                          const std::string& name, int out_idx = 0) {
  SetTensorName(layer->getOutput(out_idx), prefix, name);
}

static bool SetOutputRange(nvinfer1::ILayer* layer, float max_val, int out_idx = 0) {
  return layer->getOutput(out_idx)->setDynamicRange(-max_val, max_val);
}

//////////////////////////////////////////
//                                      //
//             Smart Pointer            //
//                                      //
//////////////////////////////////////////

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

}  // namespace TrtCommon

FWD_NAMESPACE_END
