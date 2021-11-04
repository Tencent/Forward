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
#include "trt_engine/trt_network_crt/plugins/layer_norm_plugin/layer_norm_plugin.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

FWD_TRT_NAMESPACE_BEGIN

static inline nvinfer1::DataType getParamWordType(nvinfer1::DataType cfgType) {
  if (cfgType == nvinfer1::DataType::kINT8) {
    return nvinfer1::DataType::kHALF;
  }

  return cfgType;
}

LayerNormPluginDynamic::LayerNormPluginDynamic(const std::string name,
                                               const nvinfer1::DataType type, const int ld,
                                               const nvinfer1::Weights& beta,
                                               const nvinfer1::Weights& gamma,
                                               const nvinfer1::Weights& bias)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mLd(ld),
      mType(type),
      mBiasDev(nullptr) {
  assert(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF ||
         mType == nvinfer1::DataType::kINT8);
  // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
  // mType is the plugin IO datatype, can be int8
  mCfgType = mType == nvinfer1::DataType::kINT8 ? nvinfer1::DataType::kHALF : mType;
  mParamWordsize = getElementSize(mCfgType);

  mBeta.convertAndCopy(beta, mCfgType);
  mGamma.convertAndCopy(gamma, mCfgType);

  mHasBias = (bias.values != nullptr);
  if (mHasBias) {
    mBias.convertAndCopy(bias, mCfgType);
  }
}

LayerNormPluginDynamic::LayerNormPluginDynamic(const std::string name, const void* data,
                                               size_t length)
    : mLayerName(name), mGammaDev(nullptr), mBetaDev(nullptr), mBiasDev(nullptr) {
  LOG(INFO) << "LayerNormPluginDynamic deserialize\n";

  // Deserialize in the same order as serialization
  using fwd::deserialize_value;

  deserialize_value(&data, &length, &mType);
  deserialize_value(&data, &length, &mCfgType);
  deserialize_value(&data, &length, &mLd);
  deserialize_value(&data, &length, &mHasBias);

  assert(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
  mParamWordsize = getElementSize(mCfgType);

  const char* d = static_cast<const char*>(data);
  mBeta.convertAndCopy(d, mLd, mCfgType);
  mGamma.convertAndCopy(d, mLd, mCfgType);
  if (mHasBias) {
    mBias.convertAndCopy(d, mLd, mCfgType);
  }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LayerNormPluginDynamic::clone() const noexcept {
  LOG(INFO) << "LayerNormPluginDynamic clone\n";

  auto p = new LayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma, mBias);
  p->initialize();
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs LayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  assert(nbInputs == 1);
  assert(outputIndex == 0);
  return inputs[0];
}

bool LayerNormPluginDynamic::supportsFormatCombination(int pos,
                                                       const nvinfer1::PluginTensorDesc* inOut,
                                                       int nbInputs, int nbOutputs) noexcept {
  assert(nbInputs == 1);
  assert(nbOutputs == 1);

  const nvinfer1::PluginTensorDesc& in = inOut[pos];
  if (pos == 0) {
    // Since H = W = 1, we can report CHWx for any x
    if (mType == nvinfer1::DataType::kINT8) {
      // won't work for hiddensize too small!
      nvinfer1::TensorFormat myFmt = nvinfer1::TensorFormat::kCHW32;
      if (mLd < 32) {
        myFmt = nvinfer1::TensorFormat::kCHW4;
        LOG(INFO) << "LayerNormDQQ: TensorFormat CHW4"
                  << " for LD=" << mLd << std::endl;
      } else {
        LOG(INFO) << "LayerNormDQQ: TensorFormat CHW32"
                  << " for LD=" << mLd << std::endl;
      }
      // TODO(yzx) :: do we need to check if the vectorization divides
      // mLd?
      return ((in.type == mType) && (in.format == myFmt));
    }
    return (in.type == mType) && (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  const nvinfer1::PluginTensorDesc& prev = inOut[pos - 1];

  return in.type == prev.type && in.format == prev.format;
}

void LayerNormPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                             int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc* outputs,
                                             int nbOutputs) noexcept {
  LOG(INFO) << "LayerNormPluginDynamic configurePlugin\n";

  // Validate input arguments
  assert(nbOutputs == 1);
  assert(nbInputs == 1);
  if (mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF) {
    assert(mType == inputs[0].desc.type);
  } else {
    assert(mType == inputs[0].desc.type || nvinfer1::DataType::kFLOAT == inputs[0].desc.type);
  }
  const auto& inDims0 = inputs[0].desc.dims;
  // mLd = inDims0.d[HDIM];  // hiddensize
  // assert(inDims0.nbDims == 5 || inDims0.nbDims == 3);
  mCfgType = inputs[0].desc.type == nvinfer1::DataType::kINT8 ? nvinfer1::DataType::kHALF
                                                              : inputs[0].desc.type;

  const auto paramType = getParamWordType(mCfgType);
  mParamWordsize = getElementSize(paramType);

  copyToDevice(mGamma, getWeightsSize(mGamma, paramType), mGammaDev);
  copyToDevice(mBeta, getWeightsSize(mBeta, paramType), mBetaDev);
  if (mHasBias) {
    copyToDevice(mBias, getWeightsSize(mBias, paramType), mBiasDev);
  }
}

size_t LayerNormPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                int nbInputs,
                                                const nvinfer1::PluginTensorDesc* outputs,
                                                int nbOutputs) const noexcept {
  return 0;
}

int LayerNormPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc,
                                    const void* const* inputs, void* const* outputs,
                                    void* workspace, cudaStream_t stream) noexcept {
  const int inputVolume = volume(inputDesc[0].dims);
  int status = -1;
  nvinfer1::DataType iType = inputDesc->type;

  // Our plugin outputs only one tensor
  // Launch CUDA kernel wrapper and save its return value
  if (iType == nvinfer1::DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    auto output = static_cast<float*>(outputs[0]);
    const auto bias = static_cast<const float*>(mBiasDev.get());
    const auto beta = static_cast<const float*>(mBetaDev.get());
    const auto gamma = static_cast<const float*>(mGammaDev.get());
    if (mHasBias) {
      status = computeLayerNorm<float, true>(stream, static_cast<int>(mLd), inputVolume, input,
                                             beta, gamma, output, bias);
    } else {
      status = computeLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input,
                                              beta, gamma, output, bias);
    }
  } else if (iType == nvinfer1::DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    auto output = static_cast<half*>(outputs[0]);
    const auto bias = static_cast<const half*>(mBiasDev.get());
    const auto beta = static_cast<const half*>(mBetaDev.get());
    const auto gamma = static_cast<const half*>(mGammaDev.get());
    if (mHasBias) {
      status = computeLayerNorm<half, true>(stream, static_cast<int>(mLd), inputVolume, input, beta,
                                            gamma, output, bias);
    } else {
      status = computeLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input,
                                             beta, gamma, output, bias);
    }
  } else if (iType == nvinfer1::DataType::kINT8) {
    const float dqScaleIn = inputDesc[0].scale;
    const float qScale = 1.f / outputDesc[0].scale;
    const auto input = static_cast<const int8_t*>(inputs[0]);
    auto output = static_cast<int8_t*>(outputs[0]);
    const auto bias = static_cast<const half*>(mBiasDev.get());
    const auto beta = static_cast<const half*>(mBetaDev.get());
    const auto gamma = static_cast<const half*>(mGammaDev.get());
    if (mHasBias) {
      status = computeLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, beta,
                                         gamma, output, bias, dqScaleIn, qScale);
    } else {
      status = computeLayerNormDQQ<false>(stream, static_cast<int>(mLd), inputVolume, input, beta,
                                          gamma, output, bias, dqScaleIn, qScale);
    }
  } else {
    LOG(ERROR) << "Unsupported type error, expected [kINT8,kHALF,kFLOAT], but "
                  "received "
               << static_cast<int>(iType) << "." << std::endl;
    assert(false);
  }
  return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType LayerNormPluginDynamic::getOutputDataType(int index,
                                                             const nvinfer1::DataType* inputTypes,
                                                             int nbInputs) const noexcept {
  assert(index == 0);
  assert(nbInputs == 1);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* LayerNormPluginDynamic::getPluginType() const noexcept { return FWD_LAYER_NORM_NAME; }

const char* LayerNormPluginDynamic::getPluginVersion() const noexcept {
  return FWD_LAYER_NORM_VERSION;
}

int LayerNormPluginDynamic::getNbOutputs() const noexcept { return 1; }
int LayerNormPluginDynamic::initialize() noexcept {
  LOG(INFO) << "LayerNormPluginDynamic initialize\n";
  return 0;
}

void LayerNormPluginDynamic::terminate() noexcept {
  LOG(INFO) << "LayerNormPluginDynamic terminate\n";
}

size_t LayerNormPluginDynamic::getSerializationSize() const noexcept {
  const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
  return 2 * mParamWordsize * mLd + 2 * sizeof(nvinfer1::DataType) + sizeof(mLd) + biasSize +
         sizeof(mHasBias);
}

void LayerNormPluginDynamic::serialize(void* buffer) const noexcept {
  using fwd::serialize_value;

  serialize_value(&buffer, mType);
  serialize_value(&buffer, mCfgType);
  serialize_value(&buffer, mLd);
  serialize_value(&buffer, mHasBias);

  char* d = static_cast<char*>(buffer);
  serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
  serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
  if (mHasBias) {
    serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
  }
}

void LayerNormPluginDynamic::destroy() noexcept {
  LOG(INFO) << "LayerNormPluginDynamic destroy\n";
  // This gets called when the network containing plugin is destroyed
  mGammaDev.release();
  mBetaDev.release();
  mBiasDev.release();
  mZerosDev.release();
  delete this;
}

void LayerNormPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char* LayerNormPluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

LayerNormPluginDynamicCreator::LayerNormPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* LayerNormPluginDynamicCreator::getPluginName() const noexcept {
  return FWD_LAYER_NORM_NAME;
}

const char* LayerNormPluginDynamicCreator::getPluginVersion() const noexcept {
  return FWD_LAYER_NORM_VERSION;
}

const nvinfer1::PluginFieldCollection* LayerNormPluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2* LayerNormPluginDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  LOG(INFO) << "LayerNormPluginDynamicCreator createPlugin\n";

  int ld = 0;
  int type_id = -1;
  nvinfer1::Weights beta{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights gamma{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("ld") == 0) {
      ld = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building ld: " << ld << std::endl;
    }

    if (field_name.compare("type_id") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building typeId: " << type_id << std::endl;
    }

    if (field_name.compare("beta") == 0) {
      LOG(INFO) << "Building beta...\n";
      beta.values = fc->fields[i].data;
      beta.count = fc->fields[i].length;
      beta.type = fieldTypeToDataType(fc->fields[i].type);
    }

    if (field_name.compare("gamma") == 0) {
      LOG(INFO) << "Building gamma...\n";
      gamma.values = fc->fields[i].data;
      gamma.count = fc->fields[i].length;
      gamma.type = fieldTypeToDataType(fc->fields[i].type);
    }

    if (field_name.compare("bias") == 0) {
      LOG(INFO) << "Building bias...\n";
      bias.values = fc->fields[i].data;
      bias.count = fc->fields[i].length;
      bias.type = fieldTypeToDataType(fc->fields[i].type);
    }
  }

  LOG(INFO) << "Type " << type_id << std::endl;

  if (type_id < 0 || type_id > 3) {
    LOG(ERROR) << "LayerNorm: Invalid type ID: " << type_id << std::endl;
  }

  if (beta.count <= 0 || beta.values == nullptr) {
    LOG(ERROR) << "LayerNorm: invalid beta" << std::endl;
  }

  if (gamma.count <= 0 || gamma.values == nullptr) {
    LOG(ERROR) << "LayerNorm: invalid gamma" << std::endl;
  }

  return new LayerNormPluginDynamic(name, static_cast<nvinfer1::DataType>(type_id), ld, beta, gamma,
                                    bias);
}

nvinfer1::IPluginV2* LayerNormPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call LayerNormPluginDynamic::destroy()
  return new LayerNormPluginDynamic(name, serialData, serialLength);
}

void LayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char* LayerNormPluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

FWD_TRT_NAMESPACE_END
