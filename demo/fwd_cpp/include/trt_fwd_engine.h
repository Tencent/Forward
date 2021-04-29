// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
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

#if TRT_INFER_ENABLE_PROFILING
#include <simple_profiler.h>
#endif  // TRT_INFER_ENABLE_PROFILING

#include <memory>
#include <string>
#include <vector>

#include "common/i_forward_api.h"
#include "common/trt_common.h"

FWD_NAMESPACE_BEGIN

struct TrtNetworkDesc;
class BufferManager;
class SimpleProfiler;
class EngineMetaData;

// ForwardEngine for TensorRT
class TrtForwardEngine : public IForwardEngine {
 public:
  TrtForwardEngine();

  ~TrtForwardEngine();

  // return true if forwarding succeed. The vector of outputs store results of
  // forwarding. The data ptr in Tensors can be in host memory or device memory,
  // which is determined by DeviceType. The memory of the outputs is managed
  // internally, so the caller should NOT delete the data ptr in the Tensor of
  // outputs.
  bool Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  // return true if forwarding with name succeed. The vector of outputs store
  // results of forwarding. The data ptr in Tensors can be in host memory or
  // device memory, which is determined by DeviceType. The memory of the outputs
  // is managed internally, so the caller should NOT delete the data ptr in the
  // Tensor of outputs.
  bool ForwardWithName(const IOMappingVector& inputs, IOMappingVector& outputs) override;

  // return true if engine_file is saved as engine_file.
  bool Save(const std::string& engine_file) const override;

  // return true if engine_file is loaded as engine_file.
  bool Load(const std::string& engine_file) override;

  // Clone TrtForwardEngine with Engine and MetaData.
  bool Clone(nvinfer1::ICudaEngine* engine, const EngineMetaData& meta_data);

  // return true if TensorRT engine is initialized.
  bool InitEngine();

  // return dimensions of inputs of TensorRT engine.
  std::vector<std::vector<int>> GetInputDims() const;

  // return dimensions of outputs of TensorRT engine.
  std::vector<std::vector<int>> GetOutputDims() const;

  // return DataType of the Output by the given output index.
  DataType GetOutputType(int index) const;

  // return InferMode of the Engine.
  InferMode GetMode() override;

 protected:
  // return true if TensorRT engine execute successfully. Before the execution,
  // the BATCH of engine will be set as the BATCH of inputs. After inputs and
  // outputs are all set on the device, the engine execute.
  bool Execute(const IOMappingVector& inputs, IOMappingVector& outputs);

  // return true if TensorRT engine is loaded from a saved engine_file.
  bool LoadEngine(const std::string& engine_file);

  // Check if numbers of given inputs and numbers of required inputs in the
  // Engine.
  bool CheckInputNums(std::vector<Tensor>& inputs) const;

  // Check DataTypes and Dimensions of given inputs
  bool CheckInputs(const IOMappingVector& inputs) const;

  // Set the batch size of all inputs to the given batch_size.
  bool SetBindingDimensions(int batch_size);

  // binding indices of inputs in the Engine.
  std::vector<int> input_binding_indices_;

  // Meta data of the engine.
  std::shared_ptr<EngineMetaData> meta_data_{nullptr};

  // Buffer manager to prepare INPUTs and OUTPUTs for Engine execution.
  std::shared_ptr<BufferManager> buffer_manager_{nullptr};

  // TensorRT engine.
  TrtCommon::InferUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};

  // ExecutionContext of the Engine.
  TrtCommon::InferUniquePtr<nvinfer1::IExecutionContext> context_{nullptr};

  // Cuda stream for execution.
  cudaStream_t stream_{nullptr};

#if TRT_INFER_ENABLE_PROFILING
  std::shared_ptr<utils::Profiler> profiler_{nullptr};

  std::shared_ptr<SimpleProfiler> trt_profiler_{nullptr};
#endif  // TRT_INFER_ENABLE_PROFILING
};

FWD_NAMESPACE_END
