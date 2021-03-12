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

#include "fwd_torch/torch_engine/torch_engine.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/common_macros.h"
#include "common/trt_utils.h"
#include "fwd_torch/torch_cvt/torch_helper.h"
#include "fwd_torch/torch_cvt/torch_module_parser.h"
#include "trt_engine/trt_common/trt_logger.h"
#include "trt_engine/trt_engine/trt_fwd_builder.h"
#include "trt_engine/trt_engine/trt_fwd_engine.h"

FWD_NAMESPACE_BEGIN

TorchEngine::TorchEngine(std::shared_ptr<IForwardEngine> engine) {
  engine_ = engine;
  if (engine_ == nullptr) engine_ = std::make_shared<TrtForwardEngine>();
}

TorchEngine::~TorchEngine() {}

std::vector<at::Tensor> TorchEngine::ForwardWithName(
    const std::unordered_map<std::string, c10::IValue>& input_map) const {
  IOMappingVector input_buffers;
  std::unordered_map<std::string, at::Tensor> t_input_map;
  // Parse inputs
  for (auto& entry : input_map) {
    const auto tensors = torch_::Utils::UnpackIValues({entry.second});
    for (int i = 0; i < tensors.size(); ++i) {
      const auto& name = entry.first + std::to_string(i);
      t_input_map[name] = tensors[i].toTensor();
      input_buffers.push_back({name, ParseInput(t_input_map[name])});
    }
  }

  IOMappingVector output_buffers;
  if (!engine_->ForwardWithName(input_buffers, output_buffers)) {
    return {};
  }
  std::vector<Tensor> raw_buffers;
  for (const auto& output_buffer : output_buffers) {
    raw_buffers.push_back(output_buffer.tensor);
  }

  const bool use_cuda = input_buffers.back().tensor.device_type == DeviceType::CUDA;

  return CopyFromBuffers(raw_buffers, use_cuda);
}
std::vector<at::Tensor> TorchEngine::Forward(const std::vector<torch::jit::IValue>& inputs) const {
  std::vector<at::Tensor> input_tensors = torch_::Utils::ToTensors(inputs);
  if (input_tensors.empty()) {
    LOG(ERROR) << "Input Value Error!";
    return {};
  }

  // Copy inputs
  std::vector<Tensor> input_buffers;
  for (auto& input_tensor : input_tensors) {
    input_buffers.push_back(ParseInput(input_tensor));
  }

  std::vector<Tensor> output_buffers;
  if (!engine_->Forward(input_buffers, output_buffers)) {
    return {};
  }

  // construct outputs
  const bool use_cuda = input_tensors[0].device().is_cuda();
  return CopyFromBuffers(output_buffers, use_cuda);
}

std::vector<at::Tensor> TorchEngine::Forward(const torch::jit::IValue& input) const {
  return Forward(std::vector<torch::jit::IValue>{input});
}

bool TorchEngine::Load(const std::string& engine_file) const {
  std::lock_guard<std::mutex> lock_guard(mtx);

  if (!engine_->Load(engine_file)) {
    LOG(ERROR) << "Load forward engine failed";
    return false;
  }

  return true;
}

bool TorchEngine::Save(const std::string& engine_file) const {
  if (!engine_->Save(engine_file)) {
    LOG(ERROR) << "Save forward engine failed";
    return false;
  }

  return true;
}

std::vector<std::vector<int>> TorchEngine::GetInputDims() const {
  return reinterpret_cast<TrtForwardEngine*>(engine_.get())->GetInputDims();
}

std::vector<std::vector<int>> TorchEngine::GetOutputDims() const {
  return reinterpret_cast<TrtForwardEngine*>(engine_.get())->GetOutputDims();
}

Tensor TorchEngine::ParseInput(at::Tensor& input_tensor) const {
  Tensor input_buffer;
  if (input_tensor.scalar_type() == c10::kLong) {
    input_tensor = input_tensor.to(c10::kInt);
  }
  if (input_tensor.scalar_type() == c10::kInt) {
    input_buffer.data_type = DataType::INT32;
  } else if (input_tensor.scalar_type() == c10::kHalf) {
    input_buffer.data_type = DataType::HALF;
  }

  input_buffer.device_type = input_tensor.device().is_cuda() ? DeviceType::CUDA : DeviceType::CPU;
  input_buffer.data = input_tensor.data_ptr();
  input_buffer.dims = TrtUtils::ToVector(torch_::Utils::DimsOf(input_tensor));
  return input_buffer;
}

std::vector<at::Tensor> TorchEngine::CopyFromBuffers(const std::vector<Tensor>& buffers,
                                                     bool use_cuda) const {
  std::vector<at::Tensor> outputs;
  const cudaMemcpyKind copy_kind = use_cuda ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  auto options = c10::TensorOptions()
                     .layout(::torch::kStrided)
                     .requires_grad(false)
                     .device(use_cuda ? c10::kCUDA : c10::kCPU);

  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto dtype = reinterpret_cast<TrtForwardEngine*>(engine_.get())->GetOutputType(i);
    options = options.dtype(dtype == DataType::HALF ? c10::kHalf : c10::kFloat);

    const std::vector<int>& dims = buffers[i].dims;
    std::vector<int64_t> shape(dims.begin(), dims.end());
    at::Tensor output_tensor = ::torch::empty(shape, options);
    CUDA_CHECK(
        cudaMemcpy(output_tensor.data_ptr(), buffers[i].data, output_tensor.nbytes(), copy_kind));
    outputs.push_back(output_tensor);
  }

  return outputs;
}

TorchBuilder::TorchBuilder() : mode_(InferMode::FLOAT) { builder_ = new TrtForwardBuilder(); }

TorchBuilder::~TorchBuilder() {
  if (builder_ != nullptr) {
    delete builder_;
  }
}
void TorchBuilder::SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator) const {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetCalibrator(calibrator);
}

void TorchBuilder::SetOptBatchSize(int size) const { builder_->SetOptBatchSize(size); }

bool TorchBuilder::SetInferMode(const std::string& mode) {
  mode_ = ParseInferMode(mode);
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode " << mode;
    return false;
  }
  return true;
}

void TorchBuilder::SetMaxWorkspaceSize(size_t size) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetMaxWorkspaceSize(size);
}

std::shared_ptr<TorchEngine> TorchBuilder::Build(const std::string& module_path,
                                                 const std::vector<torch::jit::IValue>& inputs) {
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode ";
    return nullptr;
  }

  torch_::Parser parser(mode_);
  if (!parser.Parse(module_path, inputs)) {
    LOG(ERROR) << "Parse torch module failed";
    return nullptr;
  }

  builder_->SetInferMode(mode_);

  auto engine = builder_->Build(parser.GetNetwork());
  if (engine == nullptr) {
    LOG(ERROR) << "Build forward engine failed";
    return nullptr;
  }

  return std::make_shared<TorchEngine>(engine);
}

std::shared_ptr<TorchEngine> TorchBuilder::Build(
    const std::string& module_path, const std::unordered_map<std::string, c10::IValue>& input_map) {
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode ";
    return nullptr;
  }

  torch_::Parser parser(mode_);
  if (!parser.Parse(module_path, input_map)) {
    LOG(ERROR) << "Parse torch module failed";
    return nullptr;
  }

  builder_->SetInferMode(mode_);

  auto engine = builder_->Build(parser.GetNetwork());
  if (engine == nullptr) {
    LOG(ERROR) << "Build forward engine failed";
    return nullptr;
  }

  return std::make_shared<TorchEngine>(engine);
}

FWD_NAMESPACE_END
