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

#include "trt_engine/trt_network_crt/plugins/normalization_plugin/normalization_plugin.h"

#include "common/trt_layer_desc.h"
#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

// #define ENABLE_NORMALIZATION_FLOAT16

FWD_TRT_NAMESPACE_BEGIN

cudnnStatus_t ConvertDataType(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      *cudnn_dtype = CUDNN_DATA_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *cudnn_dtype = CUDNN_DATA_HALF;
      break;
    default:
      return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

NormalizationPlugin::NormalizationPlugin(TrtNormalizationType type, float epsilon,
                                         const std::vector<float>& scale,
                                         const std::vector<float>& bias,
                                         nvinfer1::DataType data_type, int max_batch_size)
    : _type(type),
      _epsilon(epsilon),
      _nchan(scale.size()),
      _h_scale(scale),
      _h_bias(bias),
      _data_type(data_type),
      _initialized(false),
      max_batch_size_(max_batch_size) {
  ASSERT(scale.size() == bias.size());
}

NormalizationPlugin::NormalizationPlugin(TrtNormalizationType type, float epsilon,
                                         const std::vector<float>& scale,
                                         const std::vector<float>& bias,
                                         const std::vector<float>& mean,
                                         const std::vector<float>& var,
                                         nvinfer1::DataType data_type, int max_batch_size)
    : _type(type),
      _epsilon(epsilon),
      _nchan(scale.size()),
      _h_scale(scale),
      _h_bias(bias),
      _h_mean(mean),
      _h_var(var),
      _data_type(data_type),
      _initialized(false),
      max_batch_size_(max_batch_size) {}

NormalizationPlugin::NormalizationPlugin(void const* serialData, size_t serialLength)
    : _initialized(false) {
  deserialize_value(&serialData, &serialLength, &_type);
  deserialize_value(&serialData, &serialLength, &_epsilon);
  deserialize_value(&serialData, &serialLength, &_nchan);
  deserialize_value(&serialData, &serialLength, &_h_scale);
  deserialize_value(&serialData, &serialLength, &_h_bias);
  deserialize_value(&serialData, &serialLength, &_h_mean);
  deserialize_value(&serialData, &serialLength, &_h_var);
  deserialize_value(&serialData, &serialLength, &_data_type);
  deserialize_value(&serialData, &serialLength, &max_batch_size_);
}

NormalizationPlugin::~NormalizationPlugin() { terminate(); }

// NormalizationPlugin returns one output.
int NormalizationPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs NormalizationPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  const nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

int NormalizationPlugin::initialize() noexcept { return 0; }

void NormalizationPlugin::terminate() noexcept {
  if (!_initialized) {
    return;
  }
  CUDA_CHECK(cudaFree(_d_var));
  CUDA_CHECK(cudaFree(_d_mean));
  CUDA_CHECK(cudaFree(_d_bias));
  CUDA_CHECK(cudaFree(_d_scale));
  CUDNN_CHECK(cudnnDestroy(_cudnn_handle));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(_x_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(_b_desc));
  _initialized = false;
}

size_t NormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                             const nvinfer1::PluginTensorDesc* outputs,
                                             int nbOutputs) const noexcept {
  return 0;
}

int NormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                 const nvinfer1::PluginTensorDesc* outputDesc,
                                 const void* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
  assert(_initialized);

  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int h = input_dims.d[2];
  int w = input_dims.d[3];

  cudnnDataType_t cudnn_dtype;
#ifdef ENABLE_NORMALIZATION_FLOAT16
  CUDNN_CHECK(ConvertDataType(_data_type, &cudnn_dtype));
#else
  CUDNN_CHECK(ConvertDataType(inputDesc[0].type, &cudnn_dtype));
#endif

  switch (_type) {
    case TrtNormalizationType::BATCH_NORMALIZATION:
      CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, n, c, h, w));
      break;
    case TrtNormalizationType::INSTANCE_NORMALIZATION:
      CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1));
      CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
      break;
    case TrtNormalizationType::LAYER_NORMALIZATION:
      // TODO(Ao Li): 注意这种实现速度可能会非常慢
      getLogger()->log(nvinfer1::ILogger::Severity::kWARNING,
                       "Layer normalization plugin may very slow!");
      CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n, 1, 1));
      CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n, c * h, w));
      break;
    default:
      break;
  }

  float alpha = 1;
  float beta = 0;
  void const* x_ptr = inputs[0];
  void* y_ptr = outputs[0];
  CUDNN_CHECK(cudnnSetStream(_cudnn_handle, stream));
  // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
  //       overflows (NaNs) for fp32 data in some circumstances. The lower-
  //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
  //       acceptable.
  if (!_d_mean) {
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        _cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, _x_desc, x_ptr, _x_desc,
        y_ptr, _b_desc, _d_scale, _d_bias, 1., nullptr, nullptr, _epsilon, nullptr, nullptr));
  } else {
    // TODO(Paul Lu): 这里去掉PERSISITENT后具体性能有多大影响？
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        _cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, _x_desc, x_ptr, _x_desc, y_ptr,
        _b_desc, _d_scale, _d_bias, _d_mean, _d_var, _epsilon));
  }
  return 0;
}

size_t NormalizationPlugin::getSerializationSize() const noexcept {
  return (serialized_size(_type) + serialized_size(_epsilon) + serialized_size(_nchan) +
          serialized_size(_h_scale) + serialized_size(_h_bias) + serialized_size(_h_mean) +
          serialized_size(_h_var) + serialized_size(_data_type) + serialized_size(max_batch_size_));
}

void NormalizationPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, _type);
  serialize_value(&buffer, _epsilon);
  serialize_value(&buffer, _nchan);
  serialize_value(&buffer, _h_scale);
  serialize_value(&buffer, _h_bias);
  serialize_value(&buffer, _h_mean);
  serialize_value(&buffer, _h_var);
  serialize_value(&buffer, _data_type);
  serialize_value(&buffer, max_batch_size_);
}

bool NormalizationPlugin::supportsFormatCombination(int pos,
                                                    const nvinfer1::PluginTensorDesc* inOut,
                                                    int nbInputs, int nbOutputs) noexcept {
  ASSERT(inOut && nbInputs == 1 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));
#ifdef ENABLE_NORMALIZATION_FLOAT16
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
#else
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
#endif
}

const char* NormalizationPlugin::getPluginType() const noexcept {
  return NORMALIZATION_PLUGIN_NAME;
}

const char* NormalizationPlugin::getPluginVersion() const noexcept {
  return NORMALIZATION_PLUGIN_VERSION;
}

void NormalizationPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* NormalizationPlugin::clone() const noexcept {
  return new NormalizationPlugin{_type,   _epsilon, _h_scale,   _h_bias,
                                 _h_mean, _h_var,   _data_type, max_batch_size_};
}

// Set plugin namespace
void NormalizationPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

const char* NormalizationPlugin::getPluginNamespace() const noexcept { return mPluginNamespace; }

nvinfer1::DataType NormalizationPlugin::getOutputDataType(int index,
                                                          const nvinfer1::DataType* inputTypes,
                                                          int nbInputs) const noexcept {
  ASSERT(inputTypes && nbInputs > 0 && index == 0);
  return inputTypes[0];
}

void NormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                          const nvinfer1::DynamicPluginTensorDesc* out,
                                          int nbOutputs) noexcept {
  // for (int i = 0; i < nbInputs; i++) {
  //   _input_dims.push_back(in[i].desc.dims);
  //   for (int j = 0; j < in[i].desc.dims.nbDims; j++) {
  //     // Do not support dynamic dimensions
  //     ASSERT(in[i].desc.dims.d[j] != -1);
  //   }
  // }

  if (_initialized) {
    return;
  }
  const int n = max_batch_size_;
  const int c = in[0].desc.dims.d[1];

  assert(c == _nchan);
  const size_t nchan_bytes = _nchan * sizeof(float);
  // Note: We repeat the data for each batch entry so that we can do the full
  //       computation in a single CUDNN call in enqueue().
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_scale), n * nchan_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_bias), n * nchan_bytes));
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(
        cudaMemcpy(_d_scale + i * _nchan, _h_scale.data(), nchan_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(_d_bias + i * _nchan, _h_bias.data(), nchan_bytes, cudaMemcpyHostToDevice));
  }

  if (_h_mean.empty()) {
    _d_mean = nullptr;
    _d_var = nullptr;
  } else {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_mean), n * nchan_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_var), n * nchan_bytes));
    for (int i = 0; i < n; ++i) {
      CUDA_CHECK(
          cudaMemcpy(_d_mean + i * _nchan, _h_mean.data(), nchan_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(
          cudaMemcpy(_d_var + i * _nchan, _h_var.data(), nchan_bytes, cudaMemcpyHostToDevice));
    }
  }
  CUDNN_CHECK(cudnnCreate(&_cudnn_handle));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&_b_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&_x_desc));
  _initialized = true;
}

// NormalizationPluginCreator methods
NormalizationPluginCreator::NormalizationPluginCreator() {
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("type", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("scales", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("bias", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("running_mean", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("running_var", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("max_batch_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* NormalizationPluginCreator::getPluginName() const noexcept {
  return NORMALIZATION_PLUGIN_NAME;
}

const char* NormalizationPluginCreator::getPluginVersion() const noexcept {
  return NORMALIZATION_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* NormalizationPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt* NormalizationPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::vector<float> scaleValues;
  std::vector<float> biasValues;
  std::vector<float> meanValues;
  std::vector<float> varValues;
  float epsilon{};
  int type = 0;
  int data_type = 0;
  int max_batch_size = 0;
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "epsilon")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      epsilon = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "scales")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      const auto* data = static_cast<const float*>(fields[i].data);
      scaleValues.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "bias")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      const auto* data = static_cast<const float*>(fields[i].data);
      biasValues.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "running_mean")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      const auto* data = static_cast<const float*>(fields[i].data);
      meanValues.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "running_var")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
      const auto* data = static_cast<const float*>(fields[i].data);
      varValues.assign(data, data + fields[i].length);
    } else if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      type = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "max_batch_size")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      max_batch_size = *(static_cast<const int*>(fields[i].data));
    }
  }

  NormalizationPlugin* obj;
  if (meanValues.empty()) {
    obj = new NormalizationPlugin(static_cast<TrtNormalizationType>(type), epsilon, scaleValues,
                                  biasValues, static_cast<nvinfer1::DataType>(data_type),
                                  max_batch_size);
  } else {
    obj = new NormalizationPlugin(static_cast<TrtNormalizationType>(type), epsilon, scaleValues,
                                  biasValues, meanValues, varValues,
                                  static_cast<nvinfer1::DataType>(data_type), max_batch_size);
  }
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

nvinfer1::IPluginV2DynamicExt* NormalizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  NormalizationPlugin* obj = new NormalizationPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

FWD_TRT_NAMESPACE_END
