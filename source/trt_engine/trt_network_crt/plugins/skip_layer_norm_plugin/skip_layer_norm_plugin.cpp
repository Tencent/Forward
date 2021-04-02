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

#include <cuda.h>

#if CUDA_VERSION >= 10000
#include <NvInfer.h>

#include <cassert>
#include <cstring>
#include <vector>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"
#include "trt_engine/trt_network_crt/plugins/skip_layer_norm_plugin/skip_layer_norm_plugin.h"

namespace bert {

// Static class fields initialization
nvinfer1::PluginFieldCollection SkipLayerNormPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> SkipLayerNormPluginDynamicCreator::mPluginAttributes;

nvinfer1::PluginFieldCollection SkipLayerNormVarSeqlenPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SkipLayerNormVarSeqlenPluginCreator::mPluginAttributes;

static inline nvinfer1::DataType getParamWordType(nvinfer1::DataType cfgType) {
  if (cfgType == nvinfer1::DataType::kINT8) {
    return nvinfer1::DataType::kHALF;
  }

  return cfgType;
}

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name,
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

SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const void* data,
                                                       size_t length)
    : mLayerName(name), mGammaDev(nullptr), mBetaDev(nullptr), mBiasDev(nullptr) {
  LOG(INFO) << "SkipLayerNormPluginDynamic deserialize\n";

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
nvinfer1::IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const {
  LOG(INFO) << "SkipLayerNormPluginDynamic clone\n";

  auto p = new SkipLayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma, mBias);
  p->initialize();
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  assert(nbInputs == 2);
  assert(outputIndex == 0);
  assert(inputs[0].nbDims == inputs[1].nbDims);
  return inputs[0];
}

bool SkipLayerNormPluginDynamic::supportsFormatCombination(int pos,
                                                           const nvinfer1::PluginTensorDesc* inOut,
                                                           int nbInputs, int nbOutputs) {
  assert(nbInputs == 2);
  assert(nbOutputs == 1);

  const nvinfer1::PluginTensorDesc& in = inOut[pos];
  if (pos == 0) {
    // Since H = W = 1, we can report CHWx for any x
    if (mType == nvinfer1::DataType::kINT8) {
      // won't work for hiddensize too small!
      nvinfer1::TensorFormat myFmt = nvinfer1::TensorFormat::kCHW32;
      if (mLd < 32) {
        myFmt = nvinfer1::TensorFormat::kCHW4;
        LOG(INFO) << "SkipLayerNormDQQ: TensorFormat CHW4"
                  << " for LD=" << mLd << std::endl;
      } else {
        LOG(INFO) << "SkipLayerNormDQQ: TensorFormat CHW32"
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

void SkipLayerNormPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                                 int nbInputs,
                                                 const nvinfer1::DynamicPluginTensorDesc* outputs,
                                                 int nbOutputs) {
  LOG(INFO) << "SkipLayerNormPluginDynamic configurePlugin\n";

  // Validate input arguments
  assert(nbOutputs == 1);
  assert(nbInputs == 2);
  if (mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF) {
    assert(mType == inputs[0].desc.type);
    assert(mType == inputs[1].desc.type);
  } else {
    assert(mType == inputs[0].desc.type || nvinfer1::DataType::kFLOAT == inputs[0].desc.type);
    assert(mType == inputs[1].desc.type || nvinfer1::DataType::kFLOAT == inputs[1].desc.type);
  }
  const auto& inDims0 = inputs[0].desc.dims;
  const auto& inDims1 = inputs[1].desc.dims;
  TRT_UNUSED inDims1;
  assert(inDims0.nbDims == inDims1.nbDims);

  assert(inDims0.d[0] <= inDims1.d[0]);

  assert(std::equal(inDims0.d + 1, inDims0.d + inDims0.nbDims, inDims1.d + 1));

  // mLd = inDims0.d[HDIM];  // hiddensize

  assert(inDims0.nbDims == 5 || inDims0.nbDims == 3);
  // assert(inDims0.d[3] == 1);
  // assert(inDims0.d[4] == 1);

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

size_t SkipLayerNormPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                    int nbInputs,
                                                    const nvinfer1::PluginTensorDesc* outputs,
                                                    int nbOutputs) const {
  return 0;
}

int SkipLayerNormPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                        const nvinfer1::PluginTensorDesc* outputDesc,
                                        const void* const* inputs, void* const* outputs,
                                        void* workspace, cudaStream_t stream) {
  const int inputVolume = volume(inputDesc[0].dims);
  int status = -1;
  nvinfer1::DataType iType = inputDesc->type;

  // Our plugin outputs only one tensor
  // Launch CUDA kernel wrapper and save its return value
  if (iType == nvinfer1::DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto skip = static_cast<const float*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);
    const auto bias = static_cast<const float*>(mBiasDev.get());
    const auto beta = static_cast<const float*>(mBetaDev.get());
    const auto gamma = static_cast<const float*>(mGammaDev.get());
    if (mHasBias) {
      status = computeSkipLayerNorm<float, true>(stream, static_cast<int>(mLd), inputVolume, input,
                                                 skip, beta, gamma, output, bias);
    } else {
      status = computeSkipLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input,
                                                  skip, beta, gamma, output, bias);
    }
  } else if (iType == nvinfer1::DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto skip = static_cast<const half*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);
    const auto bias = static_cast<const half*>(mBiasDev.get());
    const auto beta = static_cast<const half*>(mBetaDev.get());
    const auto gamma = static_cast<const half*>(mGammaDev.get());
    if (mHasBias) {
      status = computeSkipLayerNorm<half, true>(stream, static_cast<int>(mLd), inputVolume, input,
                                                skip, beta, gamma, output, bias);
    } else {
      status = computeSkipLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input,
                                                 skip, beta, gamma, output, bias);
    }
  } else if (iType == nvinfer1::DataType::kINT8) {
    const float dqScaleIn = inputDesc[0].scale;
    const float dqScaleSkip = inputDesc[1].scale;
    const float qScale = 1.f / outputDesc[0].scale;
    const auto input = static_cast<const int8_t*>(inputs[0]);
    const auto skip = static_cast<const int8_t*>(inputs[1]);
    auto output = static_cast<int8_t*>(outputs[0]);
    const auto bias = static_cast<const half*>(mBiasDev.get());
    const auto beta = static_cast<const half*>(mBetaDev.get());
    const auto gamma = static_cast<const half*>(mGammaDev.get());
    if (mHasBias) {
      status =
          computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, skip,
                                        beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
    } else {
      status =
          computeSkipLayerNormDQQ<false>(stream, static_cast<int>(mLd), inputVolume, input, skip,
                                         beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
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
nvinfer1::DataType SkipLayerNormPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  assert(index == 0);
  assert(nbInputs == 2);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormPluginDynamic::getPluginType() const { return FWD_SKIP_LAYER_NORM_NAME; }

const char* SkipLayerNormPluginDynamic::getPluginVersion() const { return FWD_SKIP_LAYER_NORM_VERSION; }

int SkipLayerNormPluginDynamic::getNbOutputs() const { return 1; }
int SkipLayerNormPluginDynamic::initialize() {
  LOG(INFO) << "SkipLayerNormPluginDynamic initialize\n";
  return 0;
}

void SkipLayerNormPluginDynamic::terminate() {
  LOG(INFO) << "SkipLayerNormPluginDynamic terminate\n";
}

size_t SkipLayerNormPluginDynamic::getSerializationSize() const {
  const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
  return 2 * mParamWordsize * mLd + 2 * sizeof(nvinfer1::DataType) + sizeof(mLd) + biasSize +
         sizeof(mHasBias);
}

void SkipLayerNormPluginDynamic::serialize(void* buffer) const {
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

void SkipLayerNormPluginDynamic::destroy() {
  LOG(INFO) << "SkipLayerNormPluginDynamic destroy\n";
  // This gets called when the network containing plugin is destroyed
  mGammaDev.release();
  mBetaDev.release();
  mBiasDev.release();
  delete this;
}

void SkipLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamic::getPluginNamespace() const { return mNamespace.c_str(); }

/////////////////////////////////////////////////////////

SkipLayerNormPluginDynamicCreator::SkipLayerNormPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormPluginDynamicCreator::getPluginName() const {
  return FWD_SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginVersion() const {
  return FWD_SKIP_LAYER_NORM_VERSION;
}

const nvinfer1::PluginFieldCollection* SkipLayerNormPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  LOG(INFO) << "SkipLayerNormPluginDynamicCreator createPlugin\n";

  int ld = 0;
  nvinfer1::Weights beta{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights gamma{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  int typeId = -1;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("ld") == 0) {
      ld = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building ld: " << ld << std::endl;
    }

    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building typeId: " << typeId << std::endl;
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
  LOG(INFO) << "Type " << typeId << std::endl;

  if (typeId < 0 || typeId > 3) {
    LOG(ERROR) << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
  }

  if (beta.count <= 0 || beta.values == nullptr) {
    LOG(ERROR) << "SkipLayerNorm: invalid beta" << std::endl;
  }

  if (gamma.count <= 0 || gamma.values == nullptr) {
    LOG(ERROR) << "SkipLayerNorm: invalid gamma" << std::endl;
  }

  return new SkipLayerNormPluginDynamic(name, static_cast<nvinfer1::DataType>(typeId), ld, beta,
                                        gamma, bias);
}

nvinfer1::IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(const char* name,
                                                                          const void* serialData,
                                                                          size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call SkipLayerNormPluginDynamic::destroy()
  return new SkipLayerNormPluginDynamic(name, serialData, serialLength);
}

void SkipLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(const std::string name,
                                                           const nvinfer1::DataType type,
                                                           const nvinfer1::Weights& beta,
                                                           const nvinfer1::Weights& gamma,
                                                           const nvinfer1::Weights& bias)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mLd(beta.count),
      mType(type),
      mBiasDev(nullptr),
      mParamsOnDevice(false) {
  assert(mLd > 0);
  assert(beta.count == gamma.count);
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

SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(const std::string name, const void* data,
                                                           size_t length)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mBiasDev(nullptr),
      mParamsOnDevice(false) {
  LOG(INFO) << "SkipLayerNormVarSeqlenPlugin deserialize\n";

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
nvinfer1::IPluginV2DynamicExt* SkipLayerNormVarSeqlenPlugin::clone() const {
  LOG(INFO) << "SkipLayerNormVarSeqlenPlugin clone\n";

  auto p = new SkipLayerNormVarSeqlenPlugin(mLayerName, mType, mBeta, mGamma, mBias);
  p->initialize();
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs SkipLayerNormVarSeqlenPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  assert(nbInputs == 2);
  assert(outputIndex == 0);
  assert(inputs[0].nbDims == inputs[1].nbDims);
  return inputs[0];
}

bool SkipLayerNormVarSeqlenPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
  assert(nbInputs == 2);
  assert(nbOutputs == 1);

  const nvinfer1::PluginTensorDesc& in = inOut[pos];

  if (mType != in.type) return false;
  if (pos == 0) {
    // Since H = W = 1, we can report CHWx for any x
    if (mType == nvinfer1::DataType::kINT8) {
      // won't work for hiddensize too small!
      nvinfer1::TensorFormat myFmt = nvinfer1::TensorFormat::kCHW32;
      if (mLd < 32) {
        myFmt = nvinfer1::TensorFormat::kCHW4;
        LOG(INFO) << "SkipLayerNormDQQ: TensorFormat CHW4"
                  << " for LD=" << mLd << std::endl;
      } else {
        LOG(INFO) << "SkipLayerNormDQQ: TensorFormat CHW32"
                  << " for LD=" << mLd << std::endl;
      }
      // TODO(yzx) : do we need to check if the vectorization divides
      // mLd?
      return in.format == myFmt;
    }
    return in.format == nvinfer1::TensorFormat::kLINEAR;
  }
  const nvinfer1::PluginTensorDesc& prev = inOut[pos - 1];

  return in.format == prev.format;
}

void SkipLayerNormVarSeqlenPlugin::copyParamToDevice() {
  if (!mParamsOnDevice) {
    const auto paramType = getParamWordType(mCfgType);
    copyToDevice(mGamma, getWeightsSize(mGamma, paramType), mGammaDev);
    copyToDevice(mBeta, getWeightsSize(mBeta, paramType), mBetaDev);
    if (mHasBias) {
      copyToDevice(mBias, getWeightsSize(mBias, paramType), mBiasDev);
    }
    mParamsOnDevice = true;
  }
}

void SkipLayerNormVarSeqlenPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                                   int nbInputs,
                                                   const nvinfer1::DynamicPluginTensorDesc* outputs,
                                                   int nbOutputs) {
  // Validate input arguments
  assert(nbOutputs == 1);
  assert(nbInputs == 2);
  if (mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF) {
    assert(mType == inputs[0].desc.type);
    assert(mType == inputs[1].desc.type);
  } else {
    assert(mType == inputs[0].desc.type || nvinfer1::DataType::kFLOAT == inputs[0].desc.type);
    assert(mType == inputs[1].desc.type || nvinfer1::DataType::kFLOAT == inputs[1].desc.type);
  }
  const auto& inDims0 = inputs[0].desc.dims;
  const auto& inDims1 = inputs[1].desc.dims;
  TRT_UNUSED inDims1;
  assert(inDims0.nbDims == inDims1.nbDims);

  assert(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));

  mCfgType = inputs[0].desc.type == nvinfer1::DataType::kINT8 ? nvinfer1::DataType::kHALF
                                                              : inputs[0].desc.type;

  const auto paramType = getParamWordType(mCfgType);
  mParamWordsize = getElementSize(paramType);

  copyParamToDevice();
}

size_t SkipLayerNormVarSeqlenPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                      int nbInputs,
                                                      const nvinfer1::PluginTensorDesc* outputs,
                                                      int nbOutputs) const {
  return 0;
}

int SkipLayerNormVarSeqlenPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                          const nvinfer1::PluginTensorDesc* outputDesc,
                                          const void* const* inputs, void* const* outputs,
                                          void* workspace, cudaStream_t stream) {
  const int inputVolume = volume(inputDesc[0].dims);
  assert(inputVolume % mLd == 0 && "inconsistent dimensions");
  int status = -1;
  nvinfer1::DataType iType = inputDesc->type;

  // WAR to work with TRT6.0
  copyParamToDevice();

  // Our plugin outputs only one tensor
  // Launch CUDA kernel wrapper and save its return value
  if (iType == nvinfer1::DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto skip = static_cast<const float*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);
    const auto bias = static_cast<const float*>(mBiasDev.get());
    const auto beta = static_cast<const float*>(mBetaDev.get());
    const auto gamma = static_cast<const float*>(mGammaDev.get());
    if (mHasBias) {
      status = computeSkipLayerNorm<float, true>(stream, static_cast<int>(mLd), inputVolume, input,
                                                 skip, beta, gamma, output, bias);
    } else {
      status = computeSkipLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input,
                                                  skip, beta, gamma, output, bias);
    }
  } else if (iType == nvinfer1::DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto skip = static_cast<const half*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);
    const auto bias = static_cast<const half*>(mBiasDev.get());
    const auto beta = static_cast<const half*>(mBetaDev.get());
    const auto gamma = static_cast<const half*>(mGammaDev.get());
    if (mHasBias) {
      status = computeSkipLayerNorm<half, true>(stream, static_cast<int>(mLd), inputVolume, input,
                                                skip, beta, gamma, output, bias);
    } else {
      status = computeSkipLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input,
                                                 skip, beta, gamma, output, bias);
    }
  } else if (iType == nvinfer1::DataType::kINT8) {
    const float dqScaleIn = inputDesc[0].scale;
    const float dqScaleSkip = inputDesc[1].scale;
    const float qScale = 1.f / outputDesc[0].scale;
    const auto input = static_cast<const int8_t*>(inputs[0]);
    const auto skip = static_cast<const int8_t*>(inputs[1]);
    auto output = static_cast<int8_t*>(outputs[0]);
    const auto bias = static_cast<const half*>(mBiasDev.get());
    const auto beta = static_cast<const half*>(mBetaDev.get());
    const auto gamma = static_cast<const half*>(mGammaDev.get());
    if (mHasBias) {
      status =
          computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, skip,
                                        beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
    } else {
      status =
          computeSkipLayerNormDQQ<false>(stream, static_cast<int>(mLd), inputVolume, input, skip,
                                         beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
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
nvinfer1::DataType SkipLayerNormVarSeqlenPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  assert(index == 0);
  assert(nbInputs == 2);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* SkipLayerNormVarSeqlenPlugin::getPluginType() const { return FWD_SKIP_LAYER_NORM_NAME; }

const char* SkipLayerNormVarSeqlenPlugin::getPluginVersion() const {
  return FWD_SKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

int SkipLayerNormVarSeqlenPlugin::getNbOutputs() const { return 1; }
int SkipLayerNormVarSeqlenPlugin::initialize() {
  LOG(INFO) << "SkipLayerNormVarSeqlenPlugin initialize\n";
  return 0;
}

void SkipLayerNormVarSeqlenPlugin::terminate() {
  LOG(INFO) << "SkipLayerNormVarSeqlenPlugin terminate\n";
}

size_t SkipLayerNormVarSeqlenPlugin::getSerializationSize() const {
  const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
  return 2 * mParamWordsize * mLd + 2 * sizeof(nvinfer1::DataType) + sizeof(mLd) + biasSize +
         sizeof(mHasBias);
}

void SkipLayerNormVarSeqlenPlugin::serialize(void* buffer) const {
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

void SkipLayerNormVarSeqlenPlugin::destroy() {
  LOG(INFO) << "SkipLayerNormVarSeqlenPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
  mGammaDev.release();
  mBetaDev.release();
  mBiasDev.release();
  delete this;
}

void SkipLayerNormVarSeqlenPlugin::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* SkipLayerNormVarSeqlenPlugin::getPluginNamespace() const { return mNamespace.c_str(); }

/////////////////////////////////////////////////////////

SkipLayerNormVarSeqlenPluginCreator::SkipLayerNormVarSeqlenPluginCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* SkipLayerNormVarSeqlenPluginCreator::getPluginName() const {
  return FWD_SKIP_LAYER_NORM_NAME;
}

const char* SkipLayerNormVarSeqlenPluginCreator::getPluginVersion() const {
  return FWD_SKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
}

const nvinfer1::PluginFieldCollection* SkipLayerNormVarSeqlenPluginCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2* SkipLayerNormVarSeqlenPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  LOG(INFO) << "SkipLayerNormVarSeqlenPluginCreator createPlugin\n";

  nvinfer1::Weights beta{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights gamma{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  int typeId = -1;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
      LOG(INFO) << "Building typeId: " << typeId << std::endl;
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
  LOG(INFO) << "Type " << typeId << std::endl;

  if (typeId < 0 || typeId > 3) {
    LOG(ERROR) << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
  }

  if (beta.count <= 0 || beta.values == nullptr) {
    LOG(ERROR) << "SkipLayerNorm: invalid beta" << std::endl;
  }

  if (gamma.count <= 0 || gamma.values == nullptr) {
    LOG(ERROR) << "SkipLayerNorm: invalid gamma" << std::endl;
  }

  return new SkipLayerNormVarSeqlenPlugin(name, static_cast<nvinfer1::DataType>(typeId), beta,
                                          gamma, bias);
}

nvinfer1::IPluginV2* SkipLayerNormVarSeqlenPluginCreator::deserializePlugin(const char* name,
                                                                            const void* serialData,
                                                                            size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call SkipLayerNormVarSeqlenPlugin::destroy()
  return new SkipLayerNormVarSeqlenPlugin(name, serialData, serialLength);
}

void SkipLayerNormVarSeqlenPluginCreator::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* SkipLayerNormVarSeqlenPluginCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
}  // namespace bert

#endif  // CUDA_VERSION >= 10000
