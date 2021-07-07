# Support NEW Operators

----

- [Support NEW Operators](#support-new-operators)
  - [Add cast support](#add-cast-support)
    - [1. add creator of cast layer](#1-add-creator-of-cast-layer)
    - [2. Register LayerDescCreator](#2-register-layerdesccreator)
    - [3. Add the corresponding parser under the target framework related directory](#3-add-the-corresponding-parser-under-the-target-framework-related-directory)
    - [4. Register the corresponding parser of the operator](#4-register-the-corresponding-parser-of-the-operator)
    - [5. Build Forward Project, and check if it can be built successfully](#5-build-forward-project--and-check-if-it-can-be-built-successfully)
    - [6. Add unit_test for this new operator](#6-add-unit-test-for-this-new-operator)
  - [Add PLUGIN (Optional)](#add-plugin--optional-)
    - [1. Create operator-owned directory](#1-create-operator-owned-directory)
    - [2. Register Plugin in the CMakeLists](#2-register-plugin-in-the-cmakelists)
    - [3. Add declarations in the head file of Plugin](#3-add-declarations-in-the-head-file-of-plugin)
    - [4. Add implementations for Plugin](#4-add-implementations-for-plugin)
    - [5. Add kernel functions and device-related implementation](#5-add-kernel-functions-and-device-related-implementation)

----

> Use cast plugin as example

## Add cast support

### 1. add creator of cast layer

Refer to the following template, create `trt_cast_creator.h` in the directory `source\trt_engine\trt_network_crt\layer_creators`. Notice:

- If `TrtLayerDesc` of this layer does not exist, a new layer description should be added in the file  `source\common\trt_layer_desc.h`.

```c++
// add new TrtLayerDesc in trt_layer_desc.h
struct TrtCastDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Cast);

  nvinfer1::DataType otype;
};
```

- If PLUGINS are used, then
  - Create PLUGINS object by `nvinfer1::IPluginCreator` and `TrtCommon::InferUniquePtr`
  - Create PLUGINS_FIELDS objects by`nvinfer1::PluginField`. 

```c++
#pragma once
#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

#include <string>
#include <vector>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

BEGIN_NAMESPACE_FWD_TRT

// TRT Cast Layer Description Creator
template <>
class TLayerCreator<TrtCastDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network,
                            const TrtLayerDesc* layer_desc,
                            const ITensorVector&amp;amp; input_tensors) override {
    LOG(INFO) << "TrtCastDesc::CreateLayer";
    const TrtCastDesc* const cast_desc =
        dynamic_cast<const TrtCastDesc*>(layer_desc);
    T_CHECK(cast_desc);

    nvinfer1::ITensor* input = input_tensors[0];

    nvinfer1::DataType input_type = network->getInput(0)->getType();
    if (input_type == cast_desc->otype) return {input};

    // Create Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        CAST_PLUGIN_NAME, CAST_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("data_type", &amp;amp;input_type,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("output_type", &amp;amp;cast_desc->otype,
                            nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{
        static_cast<int>(field_data.size()), field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("cast", &amp;amp;plugin_data));

    // add the plugin to the TensorRT network
    nvinfer1::IPluginV2Layer* cast =
        network->addPluginV2(&amp;amp;input, 1, *plugin_obj);

    if (cast == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [cast] layer.";
      return {};
    }

    cast->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Cast]"))
            .c_str());
    return {cast->getOutput(0)};
  }
};

END_NAMESPACE_FWD_TRT

```

### 2. Register LayerDescCreator

- Find the Desc-Registry scope in `source\trt_engine\trt_network_crt\trt_creator_manager.cpp`, register with `trt_cast_creator.h`.
- If some PLUGINS are used, they should also be registered in Plugin-Registry scope.

For example,

```c++
...
#include "trt_engine/trt_network_crt/layer_creators/trt_cast_creator.h"
...
// Register Plugins
...
REGISTER_TENSORRT_PLUGIN(CastPluginCreator);
...
// Register LayerDesc
...
RegisterCreator<TrtCastDesc>();
...
```

### 3. Add the corresponding parser under the target framework related directory

Here PyTorch is the target framework, so `torch_cast_creator.h` is created in the directory `source\fwd_torch\torch_cvt\torch_desc_creators`.

- `input_values`: store input nodes of this operator
- `accessor`: an accessor to obtain constant arguments of this operator

```c++
#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/trt_torch_helper.h"

BEGIN_NAMESPACE_FWD_TORCH

// Torch Parser of Cast
template <>
class TLayerDescCreator<TrtCastDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const IValueAccessor* accessor) override {
    return node->kind() == c10::aten::to &amp;amp;&amp;amp;
           node->schema().overload_name() == "dtype";
  }

  std::shared_ptr<TrtLayerDesc> Create(
      const JitNode* node, const IValueAccessor* accessor,
      std::vector<const JitValue*>&amp;amp; input_values) override {
    const auto inputs = node->inputs();
    T_CHECK_GE(inputs.size(), 2);

    input_values.push_back(inputs[0]);

    auto layer_desc = std::make_shared<TrtCastDesc>();
    const auto type = accessor->Get(inputs[1]).toScalarType();
    layer_desc->otype = ST2DT_MAPPING.at(c10::toString(type));

    return layer_desc;
  }

 private:
  const std::unordered_map<const char*, nvinfer1::DataType> ST2DT_MAPPING = {
      {c10::toString(c10::ScalarType::Float), nvinfer1::DataType::kFLOAT},
      {c10::toString(c10::ScalarType::Half), nvinfer1::DataType::kHALF},
      {c10::toString(c10::ScalarType::QInt8), nvinfer1::DataType::kINT8},
      {c10::toString(c10::ScalarType::Int), nvinfer1::DataType::kINT32},
      {c10::toString(c10::ScalarType::Bool), nvinfer1::DataType::kBOOL},
  };
};
END_NAMESPACE_FWD_TORCH

```

### 4. Register the corresponding parser of the operator 

- Find the Parser-Registry scope in `source\fwd_torch\torch_cvt\torch_desc_manager.cpp`, register with `torch_cast_creator.h`

```c++
...
#include "torch_desc_creators/torch_cast_creator.h"
...

// Register Parser of CastDesc
...
RegisterCreator<TrtCastDesc>();
...
```

### 5. Build Forward Project, and check if it can be built successfully

### 6. Add unit_test for this new operator

Create unit_test file `test_cast_plugin.h` in the directory `source\unit_test`, refer to `test_tf_nodes.h` or `test_torch_nodes.h`.


## Add PLUGIN (Optional)

### 1. Create operator-owned directory

Create operator-owned directory `cast_plugin` in the directory `source\trt_engine\trt_network_crt\plugins`, and create four files as below:

- `CMakeLists.txt`: the content should be the same as that in other `xxx_plugin` directory. It is responsible to collect `*.h`, `*.cpp`, `*.cu` files in the same directory.
- `cast_plugin.h`: the head file of the Plugin.
- `cast_plugin.cpp`: the cpp file of the Plugin, including cpp implementation.
- `cast_plugin.cu`: the cu file of the Plugin, , including Device-related CUDA implementation.

### 2. Register Plugin in the CMakeLists

Register the director of Plugin in the `source\trt_engine\CMakeLists.txt`, and then CMake the project.

```txt
set(PLUGIN_LISTS
    ...
    cast_plugin
    ...
    )
```

### 3. Add declarations in the head file of Plugin

- Refer to the following template to edit the head file.
- Add CopyRight, Author infos.

```c++
#pragma once

#include <vector>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/plugin.h"

BEGIN_NAMESPACE_FWD_TRT
// For recognition in PLUGIN registry
constexpr const char* CAST_PLUGIN_NAME{"Cast_TRT"};
constexpr const char* CAST_PLUGIN_VERSION{"001"};

// declarations of device-related implementation in the .cu file
template <typename in_T, typename out_T>
void Cast(const in_T* input, out_T* output, size_t size);

// CastPlugin class, inherit from nvinfer1::IPluginV2DynamicExt 
class CastPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  CastPlugin(nvinfer1::DataType data_type, nvinfer1::DataType output_type);

  CastPlugin(void const* serialData, size_t serialLength);

  CastPlugin() = delete;

  ~CastPlugin() override;

  int getNbOutputs() const override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder&amp;amp;amp; exprBuilder) override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override;

  int32_t initialize() override;

  void terminate() override;

 private:
  nvinfer1::DataType data_type_;

  nvinfer1::DataType output_type_;

  std::string mPluginNamespace;
};


// CastPluginCreator, inherit from nvinfer1::plugin::BaseCreator
class CastPluginCreator : public nvinfer1::plugin::BaseCreator {
 public:
  CastPluginCreator();

  ~CastPluginCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection* getFieldNames() override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) override;

 private:
  nvinfer1::PluginFieldCollection mFC{};

  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

END_NAMESPACE_FWD_TRT

```

### 4. Add implementations for Plugin

Edit `cast_plugin.cpp`. 
Notice: Device-related implementations and `*.cuh` files cannot be included in this cpp file. Key functions are:

- override functions in Plugin class 
  - `Constructor`
  - `getOutputDimensions`
  - `enqueue`: define behaviors of Plugin in the runtime.
  - `supportFormatCombination`: define data structure and format of inputs and outputs
  - `serialize`, `deserialzie` and `clone`: clone behavior exists in the period of building engine.
- override functions in PluginCreator class 
  - `Constructor`
  - `createPlugin`: define how the plugin is created
  - `deserializePlugin`: define how the plugin is deconstructed

```c++
#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

#include <cuda_fp16.h>

#include <functional>
#include <numeric>
#include <string>

#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

BEGIN_NAMESPACE_FWD_TRT

CastPlugin::CastPlugin(nvinfer1::DataType data_type,
                       nvinfer1::DataType output_type)
    : data_type_(data_type), output_type_(output_type) {}

CastPlugin::CastPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&amp;amp;amp;serialData, &amp;amp;amp;serialLength, &amp;amp;amp;data_type_);
  deserialize_value(&amp;amp;amp;serialData, &amp;amp;amp;serialLength, &amp;amp;amp;output_type_);
}

CastPlugin::~CastPlugin() { terminate(); }

int CastPlugin::getNbOutputs() const { return 1; }

nvinfer1::DimsExprs CastPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder&amp;amp;amp; exprBuilder) {
  ASSERT(nbInputs == 1)
  return inputs[0];
}

size_t CastPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                    int nbInputs,
                                    const nvinfer1::PluginTensorDesc* outputs,
                                    int nbOutputs) const {
  return 0;
}

int CastPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                        const nvinfer1::PluginTensorDesc* outputDesc,
                        const void* const* inputs, void* const* outputs,
                        void* workspace, cudaStream_t stream) {
  ASSERT(output_type_ == nvinfer1::DataType::kFLOAT ||
         output_type_ == nvinfer1::DataType::kHALF);

  const auto volume = std::accumulate(
      inputDesc->dims.d, inputDesc->dims.d + inputDesc->dims.nbDims, 1,
      std::multiplies<int64_t>());

  switch (data_type_) {
    case nvinfer1::DataType::kHALF:
      Cast<half, float>(static_cast<const half*>(inputs[0]),
                        static_cast<float*>(outputs[0]), volume);
      break;
    case nvinfer1::DataType::kINT8:
      Cast<int8_t, float>(static_cast<const int8_t*>(inputs[0]),
                          static_cast<float*>(outputs[0]), volume);
      break;
    case nvinfer1::DataType::kINT32:
      Cast<int, float>(static_cast<const int*>(inputs[0]),
                       static_cast<float*>(outputs[0]), volume);
      break;
    case nvinfer1::DataType::kBOOL:
      Cast<bool, float>(static_cast<const bool*>(inputs[0]),
                        static_cast<float*>(outputs[0]), volume);
      break;
    default:
      break;
  }

  CUDA_CHECK(cudaGetLastError());

  return 0;
}

size_t CastPlugin::getSerializationSize() const {
  return serialized_size(data_type_) + serialized_size(output_type_);
}

void CastPlugin::serialize(void* buffer) const {
  serialize_value(&amp;amp;amp;buffer, data_type_);
  serialize_value(&amp;amp;amp;buffer, output_type_);
}

bool CastPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) {
  ASSERT(inOut &amp;amp;amp;&amp;amp;amp; nbInputs == 1 &amp;amp;amp;&amp;amp;amp; nbOutputs == 1 &amp;amp;amp;&amp;amp;amp;
         pos < (nbInputs + nbOutputs));
  return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &amp;amp;amp;&amp;amp;amp;
         inOut[1].type == nvinfer1::DataType::kFLOAT;
}

const char* CastPlugin::getPluginType() const { return CAST_PLUGIN_NAME; }

const char* CastPlugin::getPluginVersion() const { return CAST_PLUGIN_VERSION; }

void CastPlugin::destroy() { delete this; }

nvinfer1::IPluginV2DynamicExt* CastPlugin::clone() const {
  return new CastPlugin{data_type_, output_type_};
}

void CastPlugin::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* CastPlugin::getPluginNamespace() const {
  return mPluginNamespace.c_str();
}

nvinfer1::DataType CastPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  ASSERT(inputTypes &amp;amp;amp;&amp;amp;amp; nbInputs > 0 &amp;amp;amp;&amp;amp;amp; index == 0);
  return output_type_;
}

void CastPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                 int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc* out,
                                 int nbOutputs) {}

int32_t CastPlugin::initialize() { return 0; }

void CastPlugin::terminate() {}

CastPluginCreator::CastPluginCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "data_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "output_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* CastPluginCreator::getPluginName() const {
  return CAST_PLUGIN_NAME;
}

const char* CastPluginCreator::getPluginVersion() const {
  return CAST_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* CastPluginCreator::getFieldNames() {
  return &amp;amp;amp;mFC;
}

nvinfer1::IPluginV2DynamicExt* CastPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  int data_type{}, output_type{};
  const nvinfer1::PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "data_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      data_type = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "output_type")) {
      ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
      output_type = *(static_cast<const int*>(fields[i].data));
    } else {
      ASSERT(false);
    }
  }

  auto obj = new CastPlugin(static_cast<nvinfer1::DataType>(data_type),
                            static_cast<nvinfer1::DataType>(output_type));
  obj->setPluginNamespace(mNamespace.c_str());

  return obj;
}

nvinfer1::IPluginV2DynamicExt* CastPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  auto* obj = new CastPlugin{serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

END_NAMESPACE_FWD_TRT

```

### 5. Add kernel functions and device-related implementation

Edit `cast_plugin.cu`
Notice:

- There has be one function to be declared in `cast_plugin.h`, which has no prefix declaration of `__global__` or `__device__`. If this function is template function, then template instantiation is required in this file.

```c++
#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

#include <cuda_fp16.h>

BEGIN_NAMESPACE_FWD_TRT
template <typename in_t, typename out_t>
__global__ void CastKernel(const in_t* input, out_t* out, size_t size) {
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    out[idx] = static_cast<out_t>(input[idx]);
  }
}

// this function should exist for calling in the outer
template <typename in_T, typename out_T>
void Cast(const in_T* input, out_T* output, size_t size) {
  const int blockDim = 1024;
  const int gridDim = static_cast<int>((size + blockDim - 1) / blockDim);

  CastKernel<in_T, out_T><<<gridDim, blockDim>>>(
      static_cast<const in_T*>(input), static_cast<out_T*>(output), size);
}

// template instantiation is required
template void Cast<half, float>(const half* input, float* output, size_t size);
template void Cast<int, float>(const int* input, float* output, size_t size);
template void Cast<int8_t, float>(const int8_t* input, float* output, size_t size);
template void Cast<bool, float>(const bool* input, float* output, size_t size);
END_NAMESPACE_FWD_TRT
```
