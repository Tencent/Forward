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
//          Zhaoyi LUO (luozy63@gmail.com)

#include "fwd_onnx/onnx_engine/onnx_engine.h"

#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/fwd_common.h"
#include "common/trt_utils.h"
#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_common/trt_logger.h"
#include "trt_engine/trt_engine/trt_fwd_builder.h"
#include "trt_engine/trt_engine/trt_fwd_engine.h"

FWD_NAMESPACE_BEGIN

OnnxEngine::OnnxEngine(std::shared_ptr<IForwardEngine> engine) {
  engine_ = engine;
  if (engine_ == nullptr) {
    engine_ = std::make_shared<TrtForwardEngine>();
  }
}

OnnxEngine::~OnnxEngine() {}

bool OnnxEngine::Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
  if (!engine_->Forward(inputs, outputs)) {
    LOG(ERROR) << "OnnxEngine forward failed.";
    return false;
  }

  return true;
}

bool OnnxEngine::Load(const std::string& engine_file) const {
  if (!engine_->Load(engine_file)) {
    LOG(ERROR) << "Load forward engine failed.";
    return false;
  }

  return true;
}

bool OnnxEngine::Save(const std::string& engine_file) const {
  if (!engine_->Save(engine_file)) {
    LOG(ERROR) << "Save forward engine failed.";
    return false;
  }

  return true;
}

OnnxBuilder::OnnxBuilder() : mode_(InferMode::FLOAT) { builder_ = new TrtForwardBuilder(); }

OnnxBuilder::~OnnxBuilder() {
  if (builder_ != nullptr) {
    delete builder_;
  }
}

void OnnxBuilder::SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetCalibrator(calibrator);
}

void OnnxBuilder::SetOptBatchSize(int size) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetOptBatchSize(size);
}

int OnnxBuilder::GetOptBatchSize() const {
  return reinterpret_cast<TrtForwardBuilder*>(builder_)->GetOptBatchSize();
}

void OnnxBuilder::SetMaxBatchSize(int size) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetMaxBatchSize(size);
}

int OnnxBuilder::GetMaxBatchSize() const {
  return reinterpret_cast<TrtForwardBuilder*>(builder_)->GetMaxBatchSize();
}

bool OnnxBuilder::SetInferMode(const std::string& mode) {
  mode_ = ParseInferMode(mode);
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode." << mode;
    return false;
  }
  return true;
}

void OnnxBuilder::SetMaxWorkspaceSize(size_t size) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetMaxWorkspaceSize(size);
}

bool OnnxBuilder::SetInputType(nvinfer1::INetworkDefinition* network) const {
  if (network->getNbInputs() == 0) {
    LOG(ERROR) << "Network must have at least one input.";
    return false;
  }

  for (int32_t i = 0; i < network->getNbInputs(); ++i) {
    auto input = network->getInput(i);
    switch (input->getType()) {
      case nvinfer1::DataType::kFLOAT:
      case nvinfer1::DataType::kHALF:
        input->setType(mode_ == InferMode::FLOAT ? nvinfer1::DataType::kFLOAT
                                                 : nvinfer1::DataType::kHALF);
        break;
      case nvinfer1::DataType::kINT32:
      case nvinfer1::DataType::kINT8:
      case nvinfer1::DataType::kBOOL:
        break;
      default:
        LOG(ERROR) << "Unsupported infer mode.";
        return false;
    }
  }

  return true;
}

void OnnxBuilder::SetOutputPositions(const std::vector<int>& output_pos) {
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetOutputPositions(output_pos);
}

std::shared_ptr<OnnxEngine> OnnxBuilder::Build(const std::string& model_path) {
  if (mode_ == InferMode::INVALID) {
    LOG(ERROR) << "Unsupported inference mode.";
    return nullptr;
  }

  builder_->SetInferMode(mode_);

  size_t max_workspace_size = reinterpret_cast<TrtForwardBuilder*>(builder_)->GetMaxWorkspaceSize();
  reinterpret_cast<TrtForwardBuilder*>(builder_)->SetMaxWorkspaceSize(
      TrtCommon::ResetMaxWorkspaceSize(max_workspace_size));

  TrtCommon::InferUniquePtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(gLogger.getTRTLogger()));

  if (!builder) {
    LOG(ERROR) << "Create builder failed.";
    return nullptr;
  }

  const TrtCommon::InferUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  if (!network) {
    LOG(ERROR) << "Create network failed.";
    return nullptr;
  }

  if (!ParseModelFromFile(model_path, network.get())) {
    LOG(ERROR) << "Parse model failed.";
    return nullptr;
  }

  auto engine = BuildEngine(builder.get(), network.get());

  if (!engine) {
    LOG(ERROR) << "BuildEngine failed.";
    return nullptr;
  }

  auto trt_fwd_engine = std::make_shared<TrtForwardEngine>();

  // By default, an initiated engine will be returned.
  const auto meta_data = reinterpret_cast<TrtForwardBuilder*>(builder_)->GetEngineMetaData();
  if (!trt_fwd_engine->Clone(engine, meta_data) || !trt_fwd_engine->InitEngine()) {
    LOG(ERROR) << "Init Engine failed.";
    return nullptr;
  }

  return std::make_shared<OnnxEngine>(trt_fwd_engine);
}

bool OnnxBuilder::ParseModelFromFile(const std::string& model_path,
                                     nvinfer1::INetworkDefinition* network) const {
  auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

  if (!parser) {
    LOG(ERROR) << "Create parser failed.";
    return false;
  }

  parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

  if (parser->getNbErrors() > 0) {
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      LOG(ERROR) << parser->getError(i)->desc();
    }
    return false;
  }

  return true;
}

nvinfer1::ICudaEngine* OnnxBuilder::BuildEngine(nvinfer1::IBuilder* builder,
                                                nvinfer1::INetworkDefinition* network) {
  if (!SetInputType(network)) {
    LOG(ERROR) << "Set input data type failed.";
    return nullptr;
  }

  if (GetMaxBatchSize() < 0) {
    int dim0 = network->getInput(0)->getDimensions().d[0];
    int batch_size = dim0 == -1 ? 1 : dim0;
    SetMaxBatchSize(batch_size);
  }

  if (GetOptBatchSize() < 0) {
    SetOptBatchSize(GetMaxBatchSize());
  }

  auto dumped = reinterpret_cast<TrtForwardBuilder*>(builder_)->DumpNetwork(network);

  if (!dumped) {
    LOG(ERROR) << "Dump network failed.";
    return nullptr;
  }

  auto engine = reinterpret_cast<TrtForwardBuilder*>(builder_)->BuildEngine(builder, network);

  if (!engine) {
    LOG(ERROR) << "Build engine failed.";
    return nullptr;
  }

  SetOutputPositions(TrtCommon::GetOutputOrder(engine, network));

  return engine;
}

FWD_NAMESPACE_END
