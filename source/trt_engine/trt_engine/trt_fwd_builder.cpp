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
//          Zhaoyi LUO (luozy63@gmail.com)

#include "trt_engine/trt_engine/trt_fwd_builder.h"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/trt_utils.h"
#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_common/trt_logger.h"
#include "trt_engine/trt_engine/trt_fwd_engine.h"
#include "trt_engine/trt_network_crt/trt_network_creator.h"

FWD_NAMESPACE_BEGIN

std::shared_ptr<IForwardEngine> TrtForwardBuilder::Build(const TrtNetworkDesc& network_desc) {
  max_workspace_size_ = TrtCommon::ResetMaxWorkspaceSize(max_workspace_size_);

  TrtCommon::InferUniquePtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(gLogger.getTRTLogger()));

  if (!builder) {
    LOG(ERROR) << "Create builder failed.";
    return nullptr;
  }

  // builder->setMaxBatchSize(network_desc.batch_size);

  const TrtCommon::InferUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  if (!network) {
    LOG(ERROR) << "Create network failed.";
    return nullptr;
  }

  trt_::TrtNetworkCreator creator{network.get()};

  if (!creator.Create(network_desc)) {
    LOG(ERROR) << "Create Network failed.";
    return nullptr;
  }

  meta_data_.SetUnusedInputIndices(network_desc.unused_input_indices);
  meta_data_.SetMaxBatchSize(network_desc.batch_size);
  if (meta_data_.OptBatchSize() < 0) meta_data_.SetOptBatchSize(network_desc.batch_size);
  meta_data_.SetTorchModulePath(network_desc.torch_module_path);

  if (!DumpNetwork(network.get())) {
    LOG(ERROR) << "DumpNetwork failed.";
    return nullptr;
  }

  auto engine = BuildEngine(builder.get(), network.get());
  if (!engine) {
    LOG(ERROR) << "buildEngineWithConfig failed.";
    return nullptr;
  }

  meta_data_.SetOutputPositions(TrtCommon::GetOutputOrder(engine, network.get()));

  auto trt_fwd_engine = std::make_shared<TrtForwardEngine>();

  // By default, an initiated engine will be returned.
  if (!trt_fwd_engine->Clone(engine, meta_data_) || !trt_fwd_engine->InitEngine()) {
    LOG(ERROR) << "Init Engine failed.";
    return nullptr;
  }

  return trt_fwd_engine;
}

nvinfer1::ICudaEngine* TrtForwardBuilder::BuildEngine(nvinfer1::IBuilder* builder,
                                                      nvinfer1::INetworkDefinition* network) const {
  TrtCommon::InferUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

  if (!config) {
    LOG(ERROR) << "Create BuilderConfig failed.";
    return nullptr;
  }

  SetBuilderConfig(builder, config.get());

#ifdef USE_DYNAMIC_BATCH
  SetDynamicProfile(builder, network, config.get());
#endif  // USE_DYNAMIC_BATCH

  return builder->buildEngineWithConfig(*network, *config);
}

void TrtForwardBuilder::SetBuilderConfig(nvinfer1::IBuilder* builder,
                                         nvinfer1::IBuilderConfig* config) const {
  // Set MaxWorkSpaceSize
  config->setMaxWorkspaceSize(max_workspace_size_);
  LOG(INFO) << "maxWorkSpaceSize = " << config->getMaxWorkspaceSize();

  // config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

  // Set Infer Flag and calibrator
  const auto& mode = meta_data_.Mode();
  if (mode == InferMode::HALF) {
    LOG(INFO) << "Use FP16 mode.";

    if (!builder->platformHasFastFp16()) LOG(WARNING) << "Platform has no fast FP16.";

    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (mode == InferMode::INT8_CALIB || mode == InferMode::INT8) {
    LOG(INFO) << "Use INT8 mode.";

    if (!builder->platformHasFastInt8()) LOG(WARNING) << "Platform has no fast INT8.";

    // set float16 mode avoid non-int8 implementation layer
    // fall back to float32 mode
    if (mode == InferMode::INT8) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    if (!calibrator_) {
      LOG(ERROR) << "Failed to set int8 calibrator: You must call "
                    "SetCalibrator(nvinfer1::IInt8Calibrator&) before build "
                    "engine with INT8 mode.";
    }
    config->setInt8Calibrator(calibrator_.get());
  }

  // config->setEngineCapability(nvinfer1::EngineCapability::kSAFE_GPU);
}

bool TrtForwardBuilder::SetDynamicProfile(nvinfer1::IBuilder* builder,
                                          nvinfer1::INetworkDefinition* network,
                                          nvinfer1::IBuilderConfig* config) const {
  auto profile = builder->createOptimizationProfile();
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto* input = network->getInput(i);
    const auto inputName = input->getName();
    auto dims = input->getDimensions();
    // We do not need to check the return of setDimension and
    // setCalibrationProfile here as all dims are explicitly set
    dims.d[0] = 1;
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
    dims.d[0] = meta_data_.OptBatchSize();
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
    dims.d[0] = meta_data_.MaxBatchSize();
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
  }
  // only use one optimization profile
  config->addOptimizationProfile(profile);

#if NV_TENSORRT_MAJOR >= 7 && NV_TENSORRT_MINOR >= 1
  config->setCalibrationProfile(profile);
#endif

  return true;
}

bool TrtForwardBuilder::DumpNetwork(const nvinfer1::INetworkDefinition* network,
                                    const std::string& filename) {
  static std::map<nvinfer1::DataType, std::string> TYPE_MAP = {
      {nvinfer1::DataType::kFLOAT, "FLOAT32"},
      {nvinfer1::DataType::kHALF, "FLOAT16"},
      {nvinfer1::DataType::kINT8, "INT8"},
      {nvinfer1::DataType::kINT32, "INT32"},
  };

  std::fstream file(filename, std::ios::out);

  file << "========== network structure ==========" << std::endl;
  const std::string layer_name = "TensorRT layer name";
  const int max_name_length = std::max(static_cast<int>(layer_name.size()), 48);
  const auto old_settings = file.flags();
  const auto old_precision = file.precision();
  // Output header
  {
    file << std::setw(max_name_length) << layer_name << " ";
    file << std::setw(80) << "Input Shape"
         << " ";
    file << std::setw(80) << "Output Shape"
         << " " << std::endl;
  }
  const auto num_layers = network->getNbLayers();
  for (int i = 0; i < num_layers; ++i) {
    const auto layer = network->getLayer(i);
    file << std::setw(max_name_length) << layer->getName() << " ";
    std::string input_shapes = "[";
    for (int j = 0; j < layer->getNbInputs(); ++j) {
      auto input = layer->getInput(j);
      // RNNv2Layer and SliceLayer have optional inputs.
      if (input) {
        input_shapes += TrtUtils::ShapeStrOf(input->getDimensions());
        input_shapes += "{" + TYPE_MAP[input->getType()] + "},";
      }
    }
    input_shapes += "]";
    file << std::setw(80) << input_shapes << " ";
    std::string output_shapes = "[";
    for (int j = 0; j < layer->getNbOutputs(); ++j) {
      auto output = layer->getOutput(j);
      // RNNv2Layer has optional outputs.
      if (output) {
        if (output->getDimensions().nbDims < 0) {
          LOG(ERROR) << "Invalid layer output :" << layer->getName();
          return false;
        }
        output_shapes += TrtUtils::ShapeStrOf(output->getDimensions()) + ",";
        output_shapes += "{" + TYPE_MAP[output->getType()] + "},";
      }
    }
    output_shapes += "]";
    file << std::setw(80) << output_shapes << std::endl;
  }
  file << std::endl << "========== network inputs & outputs ==========" << std::endl;
  {
    file << std::setw(max_name_length) << "TensorRT Input/Output Name"
         << " ";
    file << std::setw(24) << "Data Type"
         << " ";
    file << std::setw(24) << "Shape" << std::endl;
  }
  for (int i = 0; i < network->getNbInputs(); ++i) {
    const auto input = network->getInput(i);
    file << std::setw(max_name_length) << input->getName() << " ";
    file << std::setw(24) << TYPE_MAP[input->getType()] << " ";
    if (input->getDimensions().nbDims < 0) {
      LOG(ERROR) << "Invalid layer output :" << input->getName();
      return false;
    }
    file << std::setw(24) << "[" << TrtUtils::ShapeStrOf(input->getDimensions()) << "]"
         << std::endl;
  }
  file << std::endl;
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    const auto output = network->getOutput(i);
    file << std::setw(max_name_length) << output->getName() << " ";
    file << std::setw(24) << TYPE_MAP[output->getType()] << " ";
    if (output->getDimensions().nbDims < 0) {
      LOG(ERROR) << "Invalid layer output :" << output->getName();
      return false;
    }
    file << std::setw(24) << "[" << TrtUtils::ShapeStrOf(output->getDimensions()) << "]"
         << std::endl;
  }
  file.flags(old_settings);
  file.precision(old_precision);
  file.close();

  return true;
}

void TrtForwardBuilder::SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator) {
  // TODO(yzx): check calibrator
  calibrator_ = calibrator;
}

void TrtForwardBuilder::SetOptBatchSize(int size) { meta_data_.SetOptBatchSize(size); }

int TrtForwardBuilder::GetOptBatchSize() const { return meta_data_.OptBatchSize(); }

void TrtForwardBuilder::SetMaxBatchSize(int size) { meta_data_.SetMaxBatchSize(size); }

int TrtForwardBuilder::GetMaxBatchSize() const { return meta_data_.MaxBatchSize(); }

void TrtForwardBuilder::SetInferMode(InferMode mode) { meta_data_.SetMode(mode); }

void TrtForwardBuilder::SetMaxWorkspaceSize(size_t size) { max_workspace_size_ = size; }

size_t TrtForwardBuilder::GetMaxWorkspaceSize() const { return max_workspace_size_; }

void TrtForwardBuilder::SetOutputPositions(const std::vector<int>& output_pos) {
  meta_data_.SetOutputPositions(output_pos);
}

EngineMetaData TrtForwardBuilder::GetEngineMetaData() const { return meta_data_; }

FWD_NAMESPACE_END
