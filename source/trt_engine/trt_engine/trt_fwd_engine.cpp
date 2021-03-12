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

#include "trt_engine/trt_engine/trt_fwd_engine.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "trt_engine/trt_common/trt_logger.h"
#include "trt_engine/trt_network_crt/trt_network_creator.h"

FWD_NAMESPACE_BEGIN

TrtForwardEngine::TrtForwardEngine(nvinfer1::ICudaEngine* engine, const EngineMetaData& meta_data)
    : meta_data_(meta_data) {
  engine_.reset(engine);
#if TRT_INFER_ENABLE_PROFILING
  profiler_ = std::make_shared<utils::Profiler>("TrtForwardEngine");
  trt_profiler_.reset(new SimpleProfiler("TensoRT performance"));
#endif  // TRT_INFER_ENABLE_PROFILING
}

TrtForwardEngine::~TrtForwardEngine() {
#if TRT_INFER_ENABLE_PROFILING
  if (profiler_ != nullptr) {
    std::cout.copyfmt(std::clog);
    profiler_->Print();
  }
#endif  // TRT_INFER_ENABLE_PROFILING
}

bool TrtForwardEngine::Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (!engine_) {
    LOG(ERROR) << "Failed to forward: You must build engine before forwarding.";
    return false;
  }

  std::vector<Tensor> real_inputs(inputs);
  if (!CheckInputNums(real_inputs)) return false;

  IOMappingVector named_inputs;
  IOMappingVector named_outputs;

  for (int i = 0; i < input_binding_indices_.size(); i++) {
    const auto& name = engine_->getBindingName(input_binding_indices_[i]);
    named_inputs.push_back({name, real_inputs[i]});
  }

  if (!ForwardWithName(named_inputs, named_outputs)) return false;

  for (auto& output : named_outputs) outputs.push_back(output.tensor);

  return true;
}

bool TrtForwardEngine::ForwardWithName(const IOMappingVector& inputs, IOMappingVector& outputs) {
  if (!engine_) {
    LOG(ERROR) << "Failed to forward: You must build engine before forwarding.";
    return false;
  }

  if (!CheckInputs(inputs)) {
    return false;
  }

  return Execute(inputs, outputs);
}

bool TrtForwardEngine::Execute(const IOMappingVector& inputs, IOMappingVector& outputs) {
  std::vector<void*> buffers(engine_->getNbBindings());

  const int batch_size = inputs[0].tensor.dims[0];

  if (!SetBindingDimensions(batch_size)) {
    LOG(ERROR) << "Failed to SetBindingDimensions: You must make sure that "
                  "allInputDimensionsSpecified.";
    return false;
  }

  if (!buffer_manager_.PrepareInputBuffer(engine_.get(), context_.get(), inputs, buffers)) {
    return false;
  }

  if (!buffer_manager_.PrepareOutputBuffer(engine_.get(), context_.get(), outputs, buffers,
                                           meta_data_.OutputPositions())) {
    return false;
  }

#ifdef TRT_INFER_ENABLE_PROFILING
  CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif

  // Forwards
  {
    UTILS_PROFILE(Forward);
    // inference
#ifdef TRT_INFER_ENABLE_PROFILING

#ifdef SUPPORT_RNN
    if (!context_->execute(engine_->getMaxBatchSize(), buffers.data())) {
#else
    if (!context_->executeV2(buffers.data())) {
#endif  // SUPPORT_RNN

#else
    if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
#endif  // TRT_INFER_ENABLE_PROFILING
      LOG(ERROR) << "Error when enqueue";
      return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

#if TRT_INFER_ENABLE_PROFILING
  std::ofstream fout("profiler.txt");
  fout << *trt_profiler_;
  // LOG(INFO) << "Profiler file has been written to profiler.txt";
  // trt_profiler_->exportJSONProfile("profiler.json");
#endif  // TRT_INFER_ENABLE_PROFILING

  return true;
}

bool TrtForwardEngine::Load(const std::string& engine_file) {
  UTILS_PROFILE(LoadEngine);

  LOG(INFO) << "Loading engine " << engine_file;

  if (!meta_data_.LoadMetaData(engine_file + ".meta")) return false;

  if (!LoadEngine(engine_file)) return false;

  return InitEngine();
}

bool TrtForwardEngine::Save(const std::string& engine_file) const {
  if (engine_ == nullptr) {
    LOG(ERROR) << "Failed to save engine: You must build engine before saving.";
    return false;
  }

  if (!meta_data_.SaveMetaData(engine_file + ".meta")) {
    return false;
  }

  nvinfer1::IHostMemory* data = engine_->serialize();
  std::ofstream engine;
  engine.open(engine_file, std::ios::binary | std::ios::out);

  if (!engine.is_open()) {
    LOG(ERROR) << "create engine file " << engine_file << " failed";
    return false;
  }
  // save engine
  engine.write(static_cast<const char*>(data->data()), data->size());
  engine.close();
  return true;
}

bool TrtForwardEngine::CheckInputNums(std::vector<Tensor>& inputs) const {
  const auto& unused_input_indices = meta_data_.UnusedInputIndices();
  const auto& num_bindings = input_binding_indices_.size();
  const auto& num_unused = unused_input_indices.size();
  const auto& num_inputs = inputs.size();

  if (num_inputs == num_bindings) return true;

  if (num_bindings + num_unused != num_inputs) {
    LOG(ERROR) << "Expect " << num_bindings << " inputs but received " << num_inputs;
    return false;
  }

  // remove unused inputs
  LOG(WARNING) << num_unused << " Unused inputs found";
  inputs.clear();
  for (size_t i = 0; i < num_inputs; ++i) {
    if (unused_input_indices.find(i) != unused_input_indices.end()) {
      LOG(WARNING) << "remove unused input tensor " << i;
    } else {
      inputs.push_back(inputs[i]);
    }
  }

  return true;
}

std::vector<std::vector<int>> TrtForwardEngine::GetInputDims() const {
  std::vector<std::vector<int>> input_dims;

  for (auto& input_index : input_binding_indices_) {
    auto dims = engine_->getBindingDimensions(input_index);
    input_dims.push_back(TrtUtils::ToVector(dims));
  }

  return input_dims;
}

std::vector<std::vector<int>> TrtForwardEngine::GetOutputDims() const {
  std::vector<std::vector<int>> output_dims;

  for (auto& output_index : meta_data_.OutputPositions()) {
    auto dims = engine_->getBindingDimensions(output_index);
    output_dims.push_back(TrtUtils::ToVector(dims));
  }

  return output_dims;
}

DataType TrtForwardEngine::GetOutputType(int index) const {
  const auto& output_bindings = meta_data_.OutputPositions();

  if (index >= output_bindings.size()) {
    LOG(ERROR) << "Expect output index  < " << output_bindings.size() << ", but got " << index;
    return {};
  }

  switch (engine_->getBindingDataType(output_bindings[index])) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::FLOAT;
    case nvinfer1::DataType::kHALF:
      return DataType::HALF;
    case nvinfer1::DataType::kINT32:
      return DataType::INT32;
    default:
      LOG(ERROR) << "Invalid output data type";
  }
  return {};
}

InferMode TrtForwardEngine::GetMode() { return meta_data_.Mode(); }

bool TrtForwardEngine::CheckInputs(const IOMappingVector& inputs) const {
  // TODO(Paul Lu): 未考虑从名字拿到冗余输入的情况

  if (inputs.size() != input_binding_indices_.size()) {
    LOG(ERROR) << "Invalid inputs : Expect " << input_binding_indices_.size() << " but received "
               << inputs.size();
    return false;
  }

  for (auto& input : inputs) {
    const int i = engine_->getBindingIndex(input.name.c_str());
    if (i < 0) {
      LOG(ERROR) << "Input cannnot be found in Engine : " << input.name;
      return false;
    }

    // check data type
    const auto dtype = input.tensor.data_type;
    const auto b_type = TrtCommon::FwdDataType(engine_->getBindingDataType(i));
    if (b_type != input.tensor.data_type) {
      LOG(ERROR) << "Invalid DataType: expect " << ToString(b_type) << " but received "
                 << ToString(dtype);
      return false;
    }

    // check dimensions
    auto binding_dims = TrtUtils::ToVector(engine_->getBindingDimensions(i));
#ifdef USE_DYNAMIC_BATCH
    binding_dims[0] = input.tensor.dims[0];
#endif  // USE_DYNAMIC_BATCH
    if (binding_dims != input.tensor.dims) {
      LOG(ERROR) << "Input dimension mismatch: got " << TrtUtils::ValueStrOf(input.tensor.dims)
                 << ", but expected " << TrtUtils::ValueStrOf(binding_dims);
      return false;
    }
  }
  return true;
}

bool TrtForwardEngine::InitEngine() {
  // create execution context
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    LOG(ERROR) << "createExecutionContext error";
    return false;
  }

#if TRT_INFER_ENABLE_PROFILING
  context_->setProfiler(trt_profiler_.get());
#endif  // TRT_INFER_ENABLE_PROFILING

  // set input_bindings
  input_binding_indices_.clear();
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    if (engine_->bindingIsInput(i)) input_binding_indices_.push_back(i);
  }

  if (!SetBindingDimensions(meta_data_.MaxBatchSize())) {
    LOG(ERROR) << "Failed to SetBindingDimensions: You must make sure that "
                  "allInputDimensionsSpecified.";
    return false;
  }

  buffer_manager_.Initialize(context_.get(), input_binding_indices_, meta_data_.OutputPositions());

  // create cuda stream
  CUDA_CHECK(cudaStreamCreate(&stream_));

  return true;
}

bool TrtForwardEngine::SetBindingDimensions(int batch_size) {
  for (auto& index : input_binding_indices_) {
    auto dims = engine_->getBindingDimensions(index);
    // 目前所有 Engine 皆要求使用 BatchFirst
    // 目前所有 Engine 只有 Batch 维度是支持动态的
    dims.d[0] = batch_size;
    if (!context_->setBindingDimensions(index, dims)) {
      LOG(ERROR) << "Invalid binding dimensions.";
      return false;
    }
  }
  return context_->allInputDimensionsSpecified();
}

bool TrtForwardEngine::LoadEngine(const std::string& engine_file) {
  std::ifstream eng_fin(engine_file, std::ios::binary);
  if (!eng_fin) {
    LOG(ERROR) << "Error opening engine file: " << engine_file;
    return false;
  }

  eng_fin.seekg(0, std::ifstream::end);
  const int64_t eng_size = eng_fin.tellg();
  eng_fin.seekg(0, std::ifstream::beg);

  std::vector<char> data(eng_size);
  eng_fin.read(data.data(), eng_size);
  if (!eng_fin) {
    LOG(ERROR) << "Error loading engine file: " << engine_file;
    return false;
  }

  TrtCommon::InferUniquePtr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(gLogger.getTRTLogger()));

  if (!runtime) {
    LOG(ERROR) << "createInferRuntime error";
    return false;
  }

  // destroy
  context_ = nullptr;
  engine_ = nullptr;

  engine_.reset(runtime->deserializeCudaEngine(data.data(), eng_size, nullptr));
  if (!engine_) {
    LOG(ERROR) << "deserializeCudaEngine error";
    return false;
  }
  return true;
}

FWD_NAMESPACE_END
