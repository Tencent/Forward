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

#include <cuda_runtime.h>
#include <easylogging++.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <numeric>
#include <vector>

#include "common/common_macros.h"
#include "common/fwd_common.h"
#include "fwd_keras/keras_engine/keras_engine.h"

namespace py = pybind11;

inline fwd::DataType GetFWDTypeFromDtype(const py::array& arr) {
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

std::vector<py::array_t<float>> KerasEngineForward(fwd::KerasEngine& engine,
                                                   const std::vector<py::array>& data) {
  std::vector<fwd::Tensor> inputs;
  std::vector<std::vector<int>> input_shapes;
  for (auto& arr : data) {
    fwd::Tensor tensor;
    tensor.data = const_cast<void*>(arr.data());
    tensor.dims.assign(arr.shape(), arr.shape() + arr.ndim());
    tensor.data_type = GetFWDTypeFromDtype(arr);
    inputs.push_back(tensor);
  }

  std::vector<fwd::Tensor> outputs;
  if (!engine.Forward(inputs, outputs)) {
    return {};
  }

  // TODO(percyyuan): so far, only support float type outputs
  std::vector<py::array_t<float>> results;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto num_ele =
        std::accumulate(outputs[i].dims.begin(), outputs[i].dims.end(), 1, std::multiplies<int>());
    const auto len = num_ele * sizeof(float);
    std::vector<float> output(len);
    CUDA_CHECK(cudaMemcpy(output.data(), outputs[i].data, len, cudaMemcpyDeviceToHost));
    results.push_back(py::array_t<float>(outputs[i].dims, output.data()));
  }
  return results;
}
