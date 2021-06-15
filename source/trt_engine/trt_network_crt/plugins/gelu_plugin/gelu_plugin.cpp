/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <NvInfer.h>

#include <cassert>
#include <cstring>
#include <vector>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"
#include "trt_engine/trt_network_crt/plugins/gelu_plugin/gelu_plugin.h"

using namespace nvinfer1;

namespace fwd {
namespace bert {

constexpr float f3_{3.0f};
constexpr float f004_{0.044715f};
constexpr float f079_{0.79788456080286535587989211986876f};
constexpr float f1_{1.0f};
constexpr float f05_{0.5f};

nvinfer1::ITensor* CreateGeluLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                                   bool use_fp16, bool use_int8) {
  if (use_int8) {
    return CreateGeluCombinattion(network, input);
  }
  return CreateGeluPlugin(network, input, use_fp16);
}

nvinfer1::ITensor* CreateGeluCombinattion(nvinfer1::INetworkDefinition* network,
                                          nvinfer1::ITensor* input) {
  const auto nbdims = input->getDimensions().nbDims;
  const Dims ref_dims = {nbdims, 1, 1, 1, 1, 1, 1, 1, 1};
  const nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;

  auto POW = network->addConstant(ref_dims, {dtype, &f3_, 1})->getOutput(0);
  auto MULTIPLY = network->addConstant(ref_dims, {dtype, &f004_, 1})->getOutput(0);
  auto SQRT = network->addConstant(ref_dims, {dtype, &f079_, 1})->getOutput(0);
  auto ONE = network->addConstant(ref_dims, {dtype, &f1_, 1})->getOutput(0);
  auto HALF = network->addConstant(ref_dims, {dtype, &f05_, 1})->getOutput(0);

  using EleWiseOp = nvinfer1::ElementWiseOperation;

  auto X_pow = network->addElementWise(*input, *POW, EleWiseOp::kPOW);
  auto X_mul = network->addElementWise(*X_pow->getOutput(0), *MULTIPLY, EleWiseOp::kPROD);
  auto X_add = network->addElementWise(*input, *X_mul->getOutput(0), EleWiseOp::kSUM);
  auto X_sqrt = network->addElementWise(*X_add->getOutput(0), *SQRT, EleWiseOp::kPROD);
  auto X_tanh = network->addActivation(*X_sqrt->getOutput(0), nvinfer1::ActivationType::kTANH);
  auto X_one = network->addElementWise(*X_tanh->getOutput(0), *ONE, EleWiseOp::kSUM);
  auto CDF = network->addElementWise(*X_one->getOutput(0), *HALF, EleWiseOp::kPROD);
  auto gelu_layer = network->addElementWise(*CDF->getOutput(0), *input, EleWiseOp::kPROD);
  fwd::TrtCommon::SetOutputRange(gelu_layer, MAX_GELU_VAL);
  return gelu_layer->getOutput(0);
}

nvinfer1::ITensor* CreateGeluPlugin(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                                    bool use_fp16) {
  auto dtype = use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;

  nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
      fwd::bert::FWD_GELU_PLUGIN_NAME, fwd::bert::FWD_GELU_PLUGIN_VERSION);
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("type_id", &dtype, nvinfer1::PluginFieldType::kINT32, 1);

  const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                    field_data.data()};
  const auto plugin_obj = fwd::TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
      creator->createPlugin("gelu", &plugin_data));

  nvinfer1::IPluginV2Layer* gelu_layer = network->addPluginV2(&input, 1, *plugin_obj);
  T_CHECK(gelu_layer);

  fwd::TrtCommon::SetOutputRange(gelu_layer, fwd::bert::MAX_GELU_VAL);
  return gelu_layer->getOutput(0);
}

GeluPluginDynamic::GeluPluginDynamic(const std::string name, const nvinfer1::DataType type,
                                     const Weights& bias)
    : mLayerName(name), mType(type), mLd(bias.count) {
  mHasBias = (bias.values != nullptr);
  if (mHasBias) {
    void* cudaMem{nullptr};
    CUDA_CHECK(cudaMalloc(&cudaMem, getWeightsSize(bias, mType)));
    CUDA_CHECK(
        cudaMemcpy(cudaMem, bias.values, getWeightsSize(bias, mType), cudaMemcpyHostToDevice));
    make_cuda_shared(mBiasDev, cudaMem);
  }
}

GeluPluginDynamic::GeluPluginDynamic(const std::string name, const void* data, size_t length)
    : mLayerName(name) {
  LOG(INFO) << "GeluPluginDynamic deserialize\n";
  using namespace fwd;

  deserialize_value(&data, &length, &mType);
  deserialize_value(&data, &length, &mLd);
  deserialize_value(&data, &length, &mHasBias);

  if (mHasBias) {
    assert(mLd > 0);
    const char* d = static_cast<const char*>(data);
    make_cuda_shared(mBiasDev, deserToDev<char>(d, mLd * getElementSize(mType)));
  }
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GeluPluginDynamic::clone() const {
  LOG(INFO) << "GeluPluginDynamic clone\n";
  auto plugin = new GeluPluginDynamic(*this);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::DimsExprs GeluPluginDynamic::getOutputDimensions(int outputIndex,
                                                           const nvinfer1::DimsExprs* inputs,
                                                           int nbInputs,
                                                           nvinfer1::IExprBuilder& exprBuilder) {
  return inputs[0];
}

bool GeluPluginDynamic::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                  int nbInputs, int nbOutputs) {
  const PluginTensorDesc& input = inOut[0];
  if (pos == 0) {
    return (input.type == mType) && (input.format == TensorFormat::kLINEAR);
  }
  if (pos == 1) {
    const PluginTensorDesc& output = inOut[1];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

void GeluPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc* out,
                                        int nbOutputs) {
  LOG(INFO) << "GeluPluginDynamic configurePlugin\n";
  assert(mType == in[0].desc.type);
}

size_t GeluPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                           const nvinfer1::PluginTensorDesc* outputs,
                                           int nbOutputs) const {
  return 0;
}
int GeluPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                               const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) {
  const int inputVolume = volume(inputDesc[0].dims);

  int status = -1;

  // Our plugin outputs only one tensor
  // Launch CUDA kernel wrapper and save its return value
  if (mType == nvinfer1::DataType::kFLOAT) {
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    if (mHasBias) {
      const float* bias = static_cast<float*>(mBiasDev.get());
      const int cols = inputVolume / mLd;
      const int rows = mLd;
      computeGeluBias(output, input, bias, rows, cols, stream);
    } else {
      status = computeGelu(stream, inputVolume, input, output);
    }
  } else if (mType == nvinfer1::DataType::kHALF) {
    const half* input = static_cast<const half*>(inputs[0]);

    half* output = static_cast<half*>(outputs[0]);

    if (mHasBias) {
      const half* bias = static_cast<half*>(mBiasDev.get());
      const int cols = inputVolume / mLd;
      const int rows = mLd;
      computeGeluBias(output, input, bias, rows, cols, stream);
    } else {
      status = computeGelu(stream, inputVolume, input, output);
    }
  } else {
    assert(false);
  }

  return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType GeluPluginDynamic::getOutputDataType(int index,
                                                        const nvinfer1::DataType* inputTypes,
                                                        int nbInputs) const {
  assert(index == 0);
  assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods

const char* GeluPluginDynamic::getPluginType() const { return FWD_GELU_PLUGIN_NAME; }

const char* GeluPluginDynamic::getPluginVersion() const { return FWD_GELU_PLUGIN_VERSION; }

int GeluPluginDynamic::getNbOutputs() const { return 1; }

int GeluPluginDynamic::initialize() {
  LOG(INFO) << "GeluPluginDynamic initalize\n";
  return 0;
}

void GeluPluginDynamic::terminate() { LOG(INFO) << "GeluPluginDynamic terminate\n"; }

size_t GeluPluginDynamic::getSerializationSize() const {
  const size_t wordSize = getElementSize(mType);
  const size_t biasSize = mHasBias ? mLd * wordSize : 0;
  return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd) + biasSize;
}

void GeluPluginDynamic::serialize(void* buffer) const {
  using namespace fwd;

  serialize_value(&buffer, mType);
  serialize_value(&buffer, mLd);
  serialize_value(&buffer, mHasBias);
  if (mHasBias) {
    assert(mLd > 0);
    char* d = static_cast<char*>(buffer);
    serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * getElementSize(mType));
  }
}

void GeluPluginDynamic::destroy() {
  LOG(INFO) << "GeluPluginDynamic destroy\n";
  // This gets called when the network containing plugin is destroyed
  mBiasDev.reset();
  delete this;
}

void GeluPluginDynamic::setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

const char* GeluPluginDynamic::getPluginNamespace() const { return mNamespace.c_str(); }

///////////////

GeluPluginDynamicCreator::GeluPluginDynamicCreator() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginDynamicCreator::getPluginName() const { return FWD_GELU_PLUGIN_NAME; }

const char* GeluPluginDynamicCreator::getPluginVersion() const { return FWD_GELU_PLUGIN_VERSION; }

const PluginFieldCollection* GeluPluginDynamicCreator::getFieldNames() { return &mFC; }

IPluginV2* GeluPluginDynamicCreator::createPlugin(const char* name,
                                                  const PluginFieldCollection* fc) {
  LOG(INFO) << "GeluPluginDynamicCreator createPlugin\n";

  Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  int typeId = -1;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("bias") == 0) {
      bias.values = fc->fields[i].data;
      bias.count = fc->fields[i].length;
      bias.type = fieldTypeToDataType(fc->fields[i].type);
    }
  }

  if (typeId < 0 || typeId > 3) {
    LOG(ERROR) << "GeluPluginDynamicCreator: invalid typeId " << typeId << std::endl;
    return nullptr;
  }

  return new GeluPluginDynamic(name, static_cast<nvinfer1::DataType>(typeId), bias);
}

IPluginV2* GeluPluginDynamicCreator::deserializePlugin(const char* name, const void* serialData,
                                                       size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call GeluPluginDynamic::destroy()
  return new GeluPluginDynamic(name, serialData, serialLength);
}

void GeluPluginDynamicCreator::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* GeluPluginDynamicCreator::getPluginNamespace() const { return mNamespace.c_str(); }

}  // namespace bert
}  // namespace fwd
