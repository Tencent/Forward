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

#include "fwd_keras/keras_engine/keras_engine.h"

#include <utility>

#include "fwd_keras/keras_cvt/trt_keras_parser.h"
#include "trt_engine/trt_engine/trt_fwd_builder.h"
#include "trt_engine/trt_engine/trt_fwd_engine.h"

FWD_NAMESPACE_BEGIN

KerasEngine::KerasEngine(std::shared_ptr<IForwardEngine> engine) {
  engine_ = engine;
  if (engine_ == nullptr) {
    engine_ = std::make_shared<TrtForwardEngine>();
  }
}

KerasEngine::~KerasEngine() {}

bool KerasEngine::Load(const std::string& engine_path) {
  if (!engine_->Load(engine_path)) {
    LOG(ERROR) << "Load forward engine failed";
    return false;
  }

  return true;
}

bool KerasEngine::Save(const std::string& engine_path) const {
  if (!engine_->Save(engine_path)) {
    LOG(ERROR) << "Save forward engine failed";
    return false;
  }

  return true;
}

bool KerasEngine::Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
  if (!engine_->Forward(inputs, outputs)) {
    LOG(ERROR) << "KerasEngine forward failed.";
    return false;
  }

  return true;
}

bool KerasEngine::Forward(const std::vector<void*>& inputs,
                          const std::vector<std::vector<int>>& input_dims,
                          std::vector<void*>& outputs, std::vector<std::vector<int>>& output_dims,
                          bool is_device) const {
  std::vector<Tensor> input_buffers;

  for (int i = 0; i < inputs.size(); ++i) {
    Tensor input;
    input.data = inputs[i];
    input.dims = input_dims[i];

    input_buffers.push_back(std::move(input));
  }

  std::vector<Tensor> output_buffers;

  if (!engine_->Forward(input_buffers, output_buffers)) {
    LOG(ERROR) << "KerasEngine forward failed.";
    return false;
  }

  for (auto& output : output_buffers) {
    outputs.push_back(output.data);
    output_dims.push_back(output.dims);
  }

  return true;
}

KerasBuilder::KerasBuilder() : mode_(InferMode::FLOAT) { builder_ = new TrtForwardBuilder(); }

KerasBuilder::~KerasBuilder() {
  if (builder_ != nullptr) {
    delete builder_;
  }
}

void KerasBuilder::SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetCalibrator(calibrator);
}

void KerasBuilder::SetOptBatchSize(int size) { builder_->SetOptBatchSize(size); }

bool KerasBuilder::SetInferMode(const std::string& mode) {
  mode_ = ParseInferMode(mode);
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode " << mode;
    return false;
  }
  return true;
}

void KerasBuilder::SetMaxWorkspaceSize(size_t size) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetMaxWorkspaceSize(size);
}

std::shared_ptr<KerasEngine> KerasBuilder::Build(const std::string& model_path, int batch_size) {
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode ";
    return nullptr;
  }

  keras_::Parser parser(mode_);

  if (!parser.Parse(model_path, batch_size)) {
    LOG(ERROR) << "Parse Keras Graph failed";
    return nullptr;
  }

  builder_->SetInferMode(mode_);

  auto engine = builder_->Build(parser.GetNetwork());

  if (engine == nullptr) {
    LOG(ERROR) << "Build forward engine failed";
    return nullptr;
  }

  return std::make_shared<KerasEngine>(engine);
}

FWD_NAMESPACE_END
