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

#include <string>
#include <vector>

#include "common/common_macros.h"
#include "common/trt_utils.h"

FWD_NAMESPACE_BEGIN

/**
 * \brief 推理引擎的 Device 缓存管理 类
 */
class BufferManager {
 public:
  BufferManager() = default;

  ~BufferManager() { Clear(); }

  // initialize the device memory of Input and Output.
  // First, clear the input and output buffers.
  // Then, malloc the device memory of Input and Output for internal use.
  bool Initialize(const nvinfer1::IExecutionContext *context,
                  const std::vector<int> &input_bindings, const std::vector<int> &output_pos) {
    // allocate input & output buffers
    Clear();

    for (auto &index : input_bindings) {
      void *temp;
      const auto dims = context->getBindingDimensions(index);
      CUDA_CHECK(cudaMalloc(&temp, sizeof(float) * TrtUtils::Volume(dims)));
      d_input_buffers_.push_back(temp);
    }

    for (auto &pos : output_pos) {
      void *temp;
      auto dims = context->getBindingDimensions(pos);
      CUDA_CHECK(cudaMalloc(&temp, sizeof(float) * TrtUtils::Volume(dims)));
      d_output_buffers_.push_back(temp);
    }

    return true;
  }

  // bind the input buffers into the corresponding pointer of Input tensor.
  // If the given Input tensor is on the CPU, then the internal pre-malloc device memory will be
  // used. And the data in the CPU-Input tensor will be transfer into the internal pre-malloc device
  // memory.
  bool PrepareInputBuffer(const nvinfer1::ICudaEngine *engine,
                          const nvinfer1::IExecutionContext *context, const IOMappingVector &inputs,
                          std::vector<void *> &buffers, cudaStream_t stream) {
    for (auto &input : inputs) {
      const int index = engine->getBindingIndex(input.name.c_str());
      const auto dims = context->getBindingDimensions(index);

      if (input.tensor.device_type == DeviceType::CPU) {
        // host -> device
        const auto dtype = engine->getBindingDataType(index);
        const int element_size = dtype == nvinfer1::DataType::kHALF ? 2 : 4;
        CUDA_CHECK(cudaMemcpyAsync(d_input_buffers_[index], input.tensor.data,
                                   TrtUtils::Volume(dims) * element_size, cudaMemcpyHostToDevice,
                                   stream));
        buffers[index] = d_input_buffers_[index];
      } else {
        buffers[index] = input.tensor.data;
      }
    }
    return true;
  }

  // bind the output buffers into the corresponding pointer of Output tensor.
  bool PrepareOutputBuffer(const nvinfer1::ICudaEngine *engine,
                           const nvinfer1::IExecutionContext *context, IOMappingVector &outputs,
                           std::vector<void *> &buffers, const std::vector<int> &output_bindings) {
    // Allocate output device memory
    for (size_t i = 0; i < output_bindings.size(); ++i) {
      const int index = output_bindings[i];
      const auto &output_dim = context->getBindingDimensions(index);

      buffers[index] = d_output_buffers_[i];

      const auto &output_name = engine->getBindingName(index);

      const DataType dtype = TrtCommon::FwdDataType(engine->getBindingDataType(index));
      outputs.push_back(
          {output_name,
           {d_output_buffers_[i], TrtUtils::ToVector(output_dim), dtype, DeviceType::CUDA}});
    }

    return true;
  }

 private:
  // Clear d_input_buffers and d_output_buffers.
  void Clear() {
    for (auto &input : d_input_buffers_) {
      CUDA_CHECK(cudaFree(input));
    }
    for (auto &output : d_output_buffers_) {
      CUDA_CHECK(cudaFree(output));
    }
    d_input_buffers_.clear();
    d_output_buffers_.clear();
  }

  // d_input_buffers are reserved for CPU-Input tensors.
  std::vector<void *> d_input_buffers_;

  // d_output_buffers are reserved for Output tensors.
  std::vector<void *> d_output_buffers_;
};

FWD_NAMESPACE_END
