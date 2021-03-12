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
#include "fwd_keras/keras_engine/keras_engine.h"

namespace py = pybind11;

std::vector<py::array_t<float>> KerasEngineForward(fwd::KerasEngine& engine,
                                                   const std::vector<py::array_t<float>>& data) {
  std::vector<void*> inputs;
  std::vector<std::vector<int>> input_shapes;
  for (auto& arr : data) {
    inputs.push_back(static_cast<void*>(const_cast<float*>(arr.data())));
    input_shapes.emplace_back(arr.shape(), arr.shape() + arr.ndim());
  }

  std::vector<void*> output_buffers;
  std::vector<std::vector<int>> output_dims;
  if (!engine.Forward(inputs, input_shapes, output_buffers, output_dims, false)) {
    return {};
  }

  std::vector<py::array_t<float>> results;
  for (size_t i = 0; i < output_buffers.size(); ++i) {
    const auto num_ele =
        std::accumulate(output_dims[i].begin(), output_dims[i].end(), 1, std::multiplies<int>());
    const auto len = num_ele * sizeof(float);
    std::vector<float> output(len);
    CUDA_CHECK(cudaMemcpy(output.data(), output_buffers[i], len, cudaMemcpyDeviceToHost));
    results.push_back(py::array_t<float>(output_dims[i], output.data()));
  }
  return results;
}
