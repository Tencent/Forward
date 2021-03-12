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
#include <simple_profiler.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_NAMESPACE_BEGIN

class TorchInfer {
 public:
  TorchInfer() {
#if TRT_INFER_ENABLE_PROFILING
    profiler_ = std::make_shared<utils::Profiler>("TorchInfer");
#endif  // TRT_INFER_ENABLE_PROFILING
  }

  ~TorchInfer() {
#if TRT_INFER_ENABLE_PROFILING
    if (profiler_ != nullptr) {
      std::cout.copyfmt(std::clog);
      profiler_->Print();
    }
#endif  // TRT_INFER_ENABLE_PROFILING
  }

  /**
   * \brief Load torch JIT model from model file path
   * \param model_path model file path
   * \return true when success
   */
  bool LoadModel(const std::string& model_path) {
    UTILS_PROFILE(LoadModel);

    try {
      module_ = ::torch::jit::load(model_path);
      module_.eval();
    } catch (const c10::Error& e) {
      LOG(ERROR) << "error when load model: " << e.msg();
      return false;
    }
    return true;
  }

  std::vector<at::Tensor> Forward(const std::vector<c10::IValue>& inputs, bool use_cuda = true) {
    if (use_cuda) module_.to(at::kCUDA);

    c10::IValue outputs;
    {
      UTILS_PROFILE(TorchForward);

      std::vector<c10::IValue> f_inputs(inputs);
      if (!torch_::Utils::RegularizeIValues(f_inputs)) return {};

      outputs = module_.forward(f_inputs);

#if TRT_INFER_ENABLE_PROFILING
      if (cudaDeviceSynchronize() != cudaSuccess) {
        LOG(ERROR) << "cudaDeviceSynchronize failed";
      }
#endif  // TRT_INFER_ENABLE_PROFILING
    }

    return torch_::Utils::ToTensors({outputs});
  }

  std::vector<at::Tensor> Forward(const std::unordered_map<std::string, c10::IValue>& input_map,
                                  bool use_cuda = true) {
    std::vector<c10::IValue> inputs;
    for (auto& input : module_.get_method("forward").graph()->inputs()) {
      auto entry = input_map.find(input->debugNameBase());
      if (entry != input_map.end()) inputs.push_back(entry->second);
    }

    return Forward(inputs, use_cuda);
  }

  torch::jit::script::Module& GetModuleCpu() {
    module_.to(at::kCPU);
    return module_;
  }

 private:
  torch::jit::script::Module module_;

#if TRT_INFER_ENABLE_PROFILING
  std::shared_ptr<utils::Profiler> profiler_{nullptr};
#endif  // TRT_INFER_ENABLE_PROFILING
};

FWD_NAMESPACE_END
