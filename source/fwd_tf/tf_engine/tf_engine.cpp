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

#include "fwd_tf/tf_engine/tf_engine.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/trt_utils.h"
#include "fwd_tf/tf_cvt/tf_graph_parser.h"
#include "trt_engine/trt_engine/trt_fwd_builder.h"
#include "trt_engine/trt_engine/trt_fwd_engine.h"

FWD_NAMESPACE_BEGIN

TfEngine::TfEngine(std::shared_ptr<IForwardEngine> engine) {
  engine_ = engine;
  if (engine_ == nullptr) {
    engine_ = std::make_shared<TrtForwardEngine>();
  }
}

TfEngine::~TfEngine() {}

Tensor TfEngine::ParseInput(TF_Tensor* input) const {
  Tensor input_buffer;
  if (TF_TensorType(input) == TF_INT32) {
    input_buffer.data_type = DataType::INT32;
  } else if (TF_TensorType(input) == TF_HALF) {
    input_buffer.data_type = DataType::HALF;
  }

  input_buffer.data = TF_TensorData(input);
  input_buffer.dims = TrtUtils::ToVector(tf_::DimsOf(input));

  return input_buffer;
}

std::vector<std::shared_ptr<TF_Tensor>> TfEngine::Forward(
    const std::vector<TF_Tensor*>& inputs) const {
  // inputs
  std::vector<Tensor> input_buffers;
  for (const auto& input : inputs) input_buffers.push_back(ParseInput(input));

  // outputs
  std::vector<Tensor> output_buffers;
  if (!engine_->Forward(input_buffers, output_buffers)) {
    return {};
  }

  return CopyFromBuffers(output_buffers);
}

std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> TfEngine::ForwardWithName(
    const std::unordered_map<std::string, TF_Tensor*>& inputs) const {
  // inputs
  IOMappingVector input_buffers;
  for (const auto& input : inputs) {
    input_buffers.push_back({input.first, ParseInput(input.second)});
  }

  // outputs
  IOMappingVector output_buffers;
  if (!engine_->ForwardWithName(input_buffers, output_buffers)) {
    return {};
  }

  std::vector<Tensor> raw_buffers;
  for (const auto& output_buffer : output_buffers) {
    raw_buffers.push_back(output_buffer.tensor);
  }

  auto raw_return = CopyFromBuffers(raw_buffers);

  std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> results;
  for (int i = 0; i < output_buffers.size(); i++) {
    results.push_back({output_buffers[i].name, raw_return[i]});
  }

  return results;
}

bool TfEngine::Load(const std::string& engine_file) const {
  if (!engine_->Load(engine_file)) {
    LOG(ERROR) << "Load forward engine failed.";
    return false;
  }

  return true;
}

bool TfEngine::Save(const std::string& engine_file) const {
  if (!engine_->Save(engine_file)) {
    LOG(ERROR) << "Save forward engine failed.";
    return false;
  }

  return true;
}

std::vector<std::shared_ptr<TF_Tensor>> TfEngine::CopyFromBuffers(
    const std::vector<Tensor>& buffers) const {
  std::vector<std::shared_ptr<TF_Tensor>> outputs;

  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto data_type =
        reinterpret_cast<TrtForwardEngine*>(engine_.get())->GetOutputType(i) == DataType::HALF
            ? TF_HALF
            : TF_FLOAT;

    std::vector<int64_t> dims_vec(buffers[i].dims.begin(), buffers[i].dims.end());

    std::shared_ptr<TF_Tensor> output = tf_::CreateEmptyTensor(data_type, dims_vec);

    CUDA_CHECK(cudaMemcpy(TF_TensorData(output.get()), buffers[i].data,
                          TF_TensorByteSize(output.get()), cudaMemcpyDeviceToHost));

    outputs.push_back(output);
  }
  return outputs;
}

std::vector<std::vector<int>> TfEngine::GetInputDims() const {
  return reinterpret_cast<TrtForwardEngine*>(engine_.get())->GetInputDims();
}

std::vector<std::vector<int>> TfEngine::GetOutputDims() const {
  return reinterpret_cast<TrtForwardEngine*>(engine_.get())->GetOutputDims();
}

TfBuilder::TfBuilder() : mode_(InferMode::FLOAT) { builder_ = new TrtForwardBuilder(); }

TfBuilder::~TfBuilder() {
  if (builder_ != nullptr) {
    delete builder_;
  }
}

void TfBuilder::SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetCalibrator(calibrator);
}

void TfBuilder::SetOptBatchSize(int size) { builder_->SetOptBatchSize(size); }

bool TfBuilder::SetInferMode(const std::string& mode) {
  mode_ = ParseInferMode(mode);
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode " << mode;
    return false;
  }
  return true;
}

void TfBuilder::SetMaxWorkspaceSize(size_t size) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetMaxWorkspaceSize(size);
}

std::shared_ptr<TfEngine> TfBuilder::Build(
    const std::string& model_path,
    const std::unordered_map<std::string, TF_Tensor*>& dummy_input_map) const {
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode ";
    return nullptr;
  }

  tf_::Parser parser(mode_);
  if (!parser.Parse(model_path, dummy_input_map)) {
    LOG(ERROR) << "Parse TF Graph failed";
    return nullptr;
  }

  builder_->SetInferMode(mode_);

  auto engine = builder_->Build(parser.GetNetwork());

  if (engine == nullptr) {
    LOG(ERROR) << "Build forward engine failed";
    return nullptr;
  }

  return std::make_shared<TfEngine>(engine);
}

FWD_NAMESPACE_END
