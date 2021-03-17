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
#include <tensorflow/c/c_api.h>

#include <memory>
#include <string>
#include <vector>

#include "fwd_tf/tf_cvt/tf_cpp_api.h"
#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_NAMESPACE_BEGIN

class TFInfer {
 public:
  TFInfer() {
#if TRT_INFER_ENABLE_PROFILING
    profiler_ = std::make_shared<utils::Profiler>("TFInfer");
#endif  // TRT_INFER_ENABLE_PROFILING
  }

  ~TFInfer() {
#if TRT_INFER_ENABLE_PROFILING
    if (profiler_ != nullptr) {
      std::cout.copyfmt(std::clog);
      profiler_->Print();
    }
#endif  // TRT_INFER_ENABLE_PROFILING
  }

  bool LoadGraph(const std::string& graph_path) {
    UTILS_PROFILE(LoadGraph);

    if (!graph_.Load(graph_path, InferMode::FLOAT)) {
      return false;
    }

    session_ = tf_::CreateSession(graph_);

    return session_ != nullptr;
  }

  bool PrepareInput(const std::unordered_map<std::string, TF_Tensor*>& input_map,
                    std::vector<TF_Output>& input_ops, std::vector<TF_Tensor*>& inputs) {
    for (const auto& entry : input_map) {
      const auto& name = entry.first;
      const auto& tensor = entry.second;
      const auto input = TF_GraphOperationByName(graph_.get(), name.c_str());
      if (input == nullptr) {
        LOG(ERROR) << "no input found : " << entry.first;
        return false;
      }
      input_ops.push_back({input, 0});
      const auto dtype = TF_OperationOutputType(input_ops.back());
      if (dtype == TF_DataType::TF_INT64 && TF_TensorType(tensor) == TF_DataType::TF_INT32) {
        auto f_tensor = tf_::CastTensor<int64_t>(tensor, TF_INT64);
        inputs.push_back(f_tensor.get());
        buffers_.push_back(f_tensor);
      } else if (TF_TensorType(tensor) == TF_DataType::TF_HALF) {
        auto f_tensor = tf_::CastTensor<float>(tensor, TF_FLOAT);
        inputs.push_back(f_tensor.get());
        buffers_.push_back(f_tensor);
      } else {
        inputs.push_back(tensor);
      }
    }
    return true;
  }

  bool PrepareOutput(const std::vector<std::string>& output_names,
                     std::vector<TF_Output>& output_ops) const {
    for (const auto& name : output_names) {
      const auto output = TF_GraphOperationByName(graph_.get(), name.c_str());
      if (output == nullptr) {
        LOG(ERROR) << "no output found : " << name;
        return false;
      }
      output_ops.push_back({output, 0});
    }
    return true;
  }

  bool Forward(const std::unordered_map<std::string, TF_Tensor*>& input_map,
               const std::vector<std::string>& output_names,
               std::vector<std::shared_ptr<TF_Tensor>>& outputs) {
    // input ops
    std::vector<TF_Output> input_ops;
    std::vector<TF_Tensor*> inputs;
    if (!PrepareInput(input_map, input_ops, inputs)) return false;

    // output ops
    std::vector<TF_Output> output_ops;
    std::vector<TF_Tensor*> output_ptrs(output_names.size(), nullptr);
    if (!PrepareOutput(output_names, output_ops)) return false;

    {
      UTILS_PROFILE(RunSession);
      tf_::Status status;
      if (!tf_::RunSession(session_.get(), input_ops.data(), inputs.data(), inputs.size(),
                                  output_ops.data(), output_ptrs.data(), output_ptrs.size(),
                                  status)) {
        LOG(ERROR) << "Error when RunSession : " << status.Message();
        return false;
      }

#if TRT_INFER_ENABLE_PROFILING
      if (cudaDeviceSynchronize() != cudaSuccess) {
        LOG(ERROR) << "cudaDeviceSynchronize failed";
      }
#endif  // TRT_INFER_ENABLE_PROFILING
    }

    // take ownership of the outputs
    outputs.clear();
    for (auto& output_ptr : output_ptrs) {
      outputs.emplace_back(output_ptr, TF_DeleteTensor);
    }

    return true;
  }

 private:
  tf_::Graph graph_;

  std::shared_ptr<TF_Session> session_;

  /**
   * \brief 用于缓冲临时生成的 TF_Tensor
   */
  std::vector<std::shared_ptr<TF_Tensor>> buffers_;

#if TRT_INFER_ENABLE_PROFILING
  std::shared_ptr<utils::Profiler> profiler_{nullptr};
#endif  // TRT_INFER_ENABLE_PROFILING
};

FWD_NAMESPACE_END
