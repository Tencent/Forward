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

#include <map>
#include <string>
#include <vector>

#include "common/common_macros.h"

// Common data structures for Forward

FWD_NAMESPACE_BEGIN

// Infer modes for Forward
enum class InferMode {
  FLOAT,
  HALF,
  INT8,
  INT8_CALIB,
  INVALID = -1,
};

// Parse inference mode from string.
inline InferMode ParseInferMode(const std::string& value) {
  if (value == "float" || value == "float32") {
    return InferMode::FLOAT;
  }
  if (value == "half" || value == "float16") {
    return InferMode::HALF;
  }
  if (value == "int8") {
    return InferMode::INT8;
  }
  if (value == "int8_calib") {
    return InferMode::INT8_CALIB;
  }

  return InferMode::INVALID;
}

// Convert inference mode to string.
inline std::string ToString(InferMode mode) {
  switch (mode) {
    case InferMode::FLOAT:
      return "float";
    case InferMode::HALF:
      return "half";
    case InferMode::INT8:
      return "int8";
    case InferMode::INT8_CALIB:
      return "int8_calib";
    default:
      return "invalid";
  }
}

// Enumerations of data type in Forward
enum class DataType {
  FLOAT,
  HALF,
  INT8,
  INT32,
  INVALID,
};

// return the string of DataType.
inline std::string ToString(DataType type) {
  switch (type) {
    case DataType::FLOAT:
      return "FLOAT";
    case DataType::HALF:
      return "HALF";
    case DataType::INT8:
      return "INT8";
    case DataType::INT32:
      return "INT32";
    default:
      return "INVALID";
  }
}

// enumerations of device types
enum class DeviceType {
  CPU,
  CUDA,
};

// Tensor in Forward stores the data pointer, dimension, DataType and DeviceType of a data tensor.
// This struct will NOT be responsible to manage the memory of the data pointer.
struct Tensor {
  void* data{nullptr};
  std::vector<int> dims;
  DataType data_type{DataType::FLOAT};
  DeviceType device_type{DeviceType::CPU};
};

// named tensor for IOMappingVector
struct NamedTensor {
  std::string name;
  Tensor tensor;
};

using IOMappingVector = std::vector<NamedTensor>;

FWD_NAMESPACE_END
