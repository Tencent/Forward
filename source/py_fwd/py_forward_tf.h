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

// #include <tf_engine.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "fwd_tf/tf_cvt/tf_utils.h"
#include "fwd_tf/tf_engine/tf_engine.h"

namespace py = pybind11;

static TF_DataType GetDataTypeFromDtype(const py::array& arr) {
  if (arr.dtype().kind() == 'f') {
    return arr.dtype().itemsize() == 4 ? TF_FLOAT : TF_HALF;
  }
  if (arr.dtype().kind() == 'i' || arr.dtype().kind() == 'u') {
    switch (arr.dtype().itemsize()) {
      case 1:
        return TF_INT8;
      case 2:
        return TF_INT16;
      case 4:
        return TF_INT32;
      case 8:
        return TF_INT64;
    }
  }
  return TF_RESOURCE;
}

std::shared_ptr<fwd::TfEngine> TfBuilderBuild(
    fwd::TfBuilder& builder, const std::string& model_path,
    const std::unordered_map<std::string, py::array>& dummy_input_map) {
  std::unordered_map<std::string, std::shared_ptr<TF_Tensor>> dummy_input_map_tmp;

  for (auto& entry : dummy_input_map) {
    auto& arr = entry.second;
    std::vector<int64_t> shape(arr.shape(), arr.shape() + arr.ndim());
    // TODO(Ao Li): 消除这里的内存拷贝
    auto input = fwd::tf_::Utils::CreateTensor(GetDataTypeFromDtype(arr), shape, arr.data());
    dummy_input_map_tmp[entry.first] = input;
  }

  std::unordered_map<std::string, TF_Tensor*> real_dummy_input;
  for (auto& entry : dummy_input_map_tmp) {
    real_dummy_input[entry.first] = entry.second.get();
  }

  return builder.Build(model_path, real_dummy_input);
}

std::vector<py::array_t<float>> TfEngineForward(fwd::TfEngine& engine,
                                                const std::vector<py::array>& data) {
  std::vector<std::shared_ptr<TF_Tensor>> inputs;
  for (auto& arr : data) {
    std::vector<int64_t> shape(arr.shape(), arr.shape() + arr.ndim());
    // TODO(Ao Li): 消除这里的内存拷贝
    auto input = fwd::tf_::Utils::CreateTensor(GetDataTypeFromDtype(arr), shape, arr.data());
    inputs.push_back(input);
  }

  std::vector<TF_Tensor*> real_inputs;
  for (auto& input : inputs) {
    real_inputs.push_back(input.get());
  }

  auto outputs = engine.Forward(real_inputs);

  std::vector<py::array_t<float>> results;
  for (auto& output : outputs) {
    auto out_shape = fwd::tf_::Utils::GetTensorShape(output.get());
    results.push_back(
        py::array_t<float>(out_shape, static_cast<const float*>(TF_TensorData(output.get()))));
  }
  return results;
}

std::unordered_map<std::string, py::array> TfEngineForwardWithName(
    fwd::TfEngine& engine, const std::unordered_map<std::string, py::array>& input_map) {
  std::unordered_map<std::string, std::shared_ptr<TF_Tensor>> input_map_tmp;

  for (auto& entry : input_map) {
    auto& arr = entry.second;
    std::vector<int64_t> shape(arr.shape(), arr.shape() + arr.ndim());
    // TODO(Ao Li): 消除这里的内存拷贝
    auto input = fwd::tf_::Utils::CreateTensor(GetDataTypeFromDtype(arr), shape, arr.data());
    input_map_tmp[entry.first] = input;
  }

  std::unordered_map<std::string, TF_Tensor*> real_inputs;
  for (auto& entry : input_map_tmp) {
    real_inputs.insert({entry.first, entry.second.get()});
  }

  auto outputs = engine.ForwardWithName(real_inputs);

  std::unordered_map<std::string, py::array> results;
  for (auto& output : outputs) {
    auto* tensor = output.second.get();
    results[output.first] = py::array_t<float>(fwd::tf_::Utils::GetTensorShape(tensor),
                                               static_cast<const float*>(TF_TensorData(tensor)));
  }
  return results;
}
