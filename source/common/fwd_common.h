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

#include <string>
#include <unordered_map>
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
  const std::unordered_map<std::string, InferMode> STR_MODE_MAP = {
      {"float", InferMode::FLOAT}, {"float32", InferMode::FLOAT},
      {"half", InferMode::HALF},   {"float16", InferMode::HALF},
      {"int8", InferMode::INT8},   {"int8_calib", InferMode::INT8_CALIB},
  };
  const auto mode = STR_MODE_MAP.find(value);
  if (mode == STR_MODE_MAP.end()) {
    return InferMode::INVALID;
  }
  return mode->second;
}

// Enumerations of data type in Forward
enum class DataType {
  FLOAT,
  HALF,
  INT8,
  INT16,
  INT32,
  INT64,
  DOUBLE,
  INVALID,
};

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
