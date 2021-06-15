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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fwd_torch/torch_cvt/torch_helper.h"
#include "fwd_torch/torch_engine/torch_engine.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <>
struct type_caster<torch::jit::IValue> {
 public:
  PYBIND11_TYPE_CASTER(torch::jit::IValue, _("IValue"));

  bool load(handle src, bool) {
    try {
      value = torch::jit::toTypeInferredIValue(src);
      return true;
    } catch (std::exception& e) {
      return false;
    }
  }

  static handle cast(torch::jit::IValue src, return_value_policy /* policy */,
                     handle /* parent */) {
    return torch::jit::toPyObject(std::move(src)).release();
  }
};
}  // namespace detail
}  // namespace pybind11

std::shared_ptr<fwd::TorchEngine> TorchBuilderBuild(fwd::TorchBuilder& builder,
                                                    const std::string& module_path,
                                                    const std::vector<torch::jit::IValue>& inputs) {
  return builder.Build(module_path, inputs);
}

std::shared_ptr<fwd::TorchEngine> TorchBuilderBuild(fwd::TorchBuilder& builder,
                                                    const std::string& module_path,
                                                    const torch::jit::IValue& input) {
  return builder.Build(module_path, {input});
}

std::shared_ptr<fwd::TorchEngine> TorchBuilderBuildWithName(
    fwd::TorchBuilder& builder, const std::string& module_path,
    const std::unordered_map<std::string, c10::IValue>& input_map) {
  return builder.Build(module_path, input_map);
}

torch::jit::IValue TorchEngineForward(fwd::TorchEngine& engine,
                                      const std::vector<torch::jit::IValue>& input) {
  const auto results = engine.Forward(input);
  if (results.empty()) {
    return {};
  }
  if (results.size() == 1) {
    return results[0];
  }
  return results;
}

torch::jit::IValue TorchEngineForward(fwd::TorchEngine& engine, const torch::jit::IValue& input) {
  return TorchEngineForward(engine, std::vector<torch::jit::IValue>{input});
}

torch::jit::IValue TorchEngineForwardWithName(
    fwd::TorchEngine& engine, const std::unordered_map<std::string, c10::IValue>& input_map) {
  const auto results = engine.ForwardWithName(input_map);
  if (results.empty()) {
    return {};
  }
  if (results.size() == 1) {
    return results[0];
  }
  return results;
}
