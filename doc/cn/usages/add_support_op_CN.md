# 开源共建：扩展添加支持操作的流程

----

> 本流程以 cast plugin 为例

## 添加支持操作层级

### 1. 添加操作层级创建器

在 `source\trt_engine\trt_network_crt\layer_creators` 目录下创建使用层级创建器文件 `trt_cast_creator.h` 。参照以下模板创建层级创建器。须注意：

- 如果是目前项目暂未支持的操作，需要添加一个新的操作描述：**新描述添加须要提 Issue**。 在 `source\common\trt_layer_desc.h` 中添加描述。

```c++
// 项目组自定义的描述
struct TrtCastDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Cast);

  nvinfer1::DataType otype;
};
```

- 尽量使用项目本身支持的描述，以及 TensorRT 原生层级 `ILayer` 进行支持。
- 如果需要使用插件支持
  - 使用正确的标识名及版本号创建 `nvinfer1::IPluginCreator` ；
  - 使用正确的参数名, 参数类型及大小来创建 `nvinfer1::PluginField` 集合；
  - 使用 `TrtCommon::InferUniquePtr` 创建 `plugin_obj` 。

```c++
#pragma once
#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

#include <string>
#include <vector>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

BEGIN_NAMESPACE_FWD_TRT

// TRT Cast 层描述创建器
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

    // 创建 Plugin
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

### 2. 注册层级创建器

在 `source\trt_engine\trt_network_crt\trt_creator_manager.cpp` 中找到层级注册区域，进行层级注册。如需要插件，也须注册插件创造器。

- 添加头文件 `trt_cast_creator.h` ；
- 如果该层级需要插件的话, 须注册插件创造器。

```c++
...
#include "trt_engine/trt_network_crt/layer_creators/trt_cast_creator.h"
...
// 注册插件创建器
...
REGISTER_TENSORRT_PLUGIN(CastPluginCreator);
...
// 注册层级创建器
...
RegisterCreator<TrtCastDesc>();
...
```

### 3. 在对应的模型框架目录下添加操作的解析

此处以 torch 模型操作为例，在 `source\fwd_torch\torch_cvt\torch_desc_creators` 目录下创建添加 `torch_cast_creator.h` 。

- 需要把输入节点放进 `input_values` ；
- 从 `accessor` 中获取模型中常量参数。

```c++
#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/trt_torch_helper.h"

BEGIN_NAMESPACE_FWD_TORCH

// Cast 层描述器
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

### 4. 注册模型操作解析器

在 `source\fwd_torch\torch_cvt\torch_desc_manager.cpp` 中找到解析器注册区域，进行解析器注册。

- 添加头文件 `torch_cast_creator.h` ；
- 注册操作解析器。

```c++
...
#include "torch_desc_creators/torch_cast_creator.h"
...

// 注册操作解析器
...
RegisterCreator<TrtCastDesc>();
...
```

### 5. 编译项目, 检查编译是否通过

### 6. 添加单元测试, 检验插件的正确性

在 `source\unit_test` 目录下添加一个单元测试文件 `test_cast_plugin.h` 。单元测试可参考 `test_tf_nodes.h` 与 `test_torch_nodes.h` 。


## 添加插件（可选）

### 1. 创建 Plugin 专用目录及相关文件

在 `source\trt_engine\trt_network_crt\plugins` 目录下创建一个专用目录 `cast_plugin`，在 `cast_plugin` 目录下创建如下 4 个文件：

- `CMakeLists.txt`：与其他 plugin 下的 `CMakeLists.txt` 的内容完全一致，用于收集同目录下的 `*.h`，`*.cpp`，`*.cu` 文件。 
- `cast_plugin.h`：Plugin 的头文件，用于声明插件类的相关类及函数。
- `cast_plugin.cpp`：Plugin 的 Cpp 代码文件，用于定义及存放插件中的与设备代码无关的 Cpp 代码。
- `cast_plugin.cu`：Plugin 的 Cuda 代码文件，用于定义及存放插件中的核函数及相关设备（Device）代码。

### 2. 在 CMakeLists 中注册 Plugin 目录，此时可以用 CMake 创建项目（可选）

在 `source\trt_engine\CMakeLists.txt` 中，注册刚才创建的插件目录。
```
set(PLUGIN_LISTS
    ...
    cast_plugin
    ...
    )
```

### 3. 在头文件中添加 Plugin 声明

打开 `cast_plugin.h` 进行编辑，参考如下模板进行自定义插件声明，需要添加 CopyRight 及 Author。

```c++
#pragma once

#include <vector>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/plugins/common/plugin.h"

BEGIN_NAMESPACE_FWD_TRT
// 插件标识名及版本号, 用于调用时识别
constexpr const char* CAST_PLUGIN_NAME{"Cast_TRT"};
constexpr const char* CAST_PLUGIN_VERSION{"001"};

// 相关的执行函数声明, 该函数的定义须存在于 *.cu 中而非 *.cpp 中. 
template <typename in_T, typename out_T>
void Cast(const in_T* input, out_T* output, size_t size);

// Plugin 类, 基本都是需要定义的各种 override 声明
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


// Plugin 创建器声明, 基本是需要定义的 Override 声明
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

### 4. 添加插件定义

打开 `cast_plugin.cpp` 进行 Plugin 方法定义重写。本文件不能包含 `*.cuh` 头文件及设备代码。关键函数为：

- 插件类
  - 构造函数
  - `getOutputDimensions`：用于确定输出维度；
  - `enqueue`：定义插件如何执行计算；
  - `supportFormatCombination`：用于定义插件支持的输入输出的数据结构及数据类型；
  - `serialize`，`deserialzie` 及 `clone`：插件在构建过程中存在 clone 操作，须定义好相关数据的序列化与反序列化。
- 插件创造类
  - 构造函数；
  - `createPlugin`：定义插件如何创造，如何接收参数；
  - `deserializePlugin`：定义插件如何析构。

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

### 5. 添加插件核函数及相关设备代码

打开 `cast_plugin.cu` 文件，定义插件中使用的核函数及相关设备代码。须注意的是：

- 必须有一个不带 `__global__`，`__device__` 的执行调用函数，该调用函数声明于 `cast_plugin.h` 中。
- 若上述的调用函数为模板函数，则需要在本文件内进行模板实例化声明。

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

// 必须存在这样一个可供 cpp 调用的函数
template <typename in_T, typename out_T>
void Cast(const in_T* input, out_T* output, size_t size) {
  const int blockDim = 1024;
  const int gridDim = static_cast<int>((size + blockDim - 1) / blockDim);

  CastKernel<in_T, out_T><<<gridDim, blockDim>>>(
      static_cast<const in_T*>(input), static_cast<out_T*>(output), size);
}

// 须进行模板实例化声明
template void Cast<half, float>(const half* input, float* output, size_t size);
template void Cast<int, float>(const int* input, float* output, size_t size);
template void Cast<int8_t, float>(const int8_t* input, float* output, size_t size);
template void Cast<bool, float>(const bool* input, float* output, size_t size);
END_NAMESPACE_FWD_TRT
```
