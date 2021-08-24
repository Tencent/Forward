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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <pybind11/numpy.h>

#include "common/fwd_common.h"

namespace py = pybind11;

// This is used to support pybind11 for Keras and ONNX models.
static fwd::DataType GetFWDTypeFromDtype(const py::array& arr) {
  if (arr.dtype().kind() == 'f') {
    switch (arr.dtype().itemsize()) {
      case 2:
        return fwd::DataType::HALF;
      case 4:
        return fwd::DataType::FLOAT;
      case 8:
        return fwd::DataType::DOUBLE;
    }
  }
  if (arr.dtype().kind() == 'i' || arr.dtype().kind() == 'u') {
    switch (arr.dtype().itemsize()) {
      case 1:
        return fwd::DataType::INT8;
      case 4:
        return fwd::DataType::INT32;
      case 8:
        return fwd::DataType::INT64;
      default:
        return fwd::DataType::INVALID;
    }
  }
  return fwd::DataType::INVALID;
}
