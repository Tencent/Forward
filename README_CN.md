![](doc/img/forward_logo_1.png)

# Forward 深度学习推理加速框架

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Build Status](https://travis-ci.com/Tencent/Forward.svg?branch=master)](https://travis-ci.com/Tencent/Forward)

----

[[English Version](README.md)]

## 简介

Forward 是一款腾讯研发的 GPU 高性能推理加速框架。它提出了一种解析方案，可直接加载主流框架模型（Tensorflow / PyTorch / Keras）转换成 TensorRT 推理加速引擎，帮助用户节省中间繁杂的模型转换或网络构建步骤。相对于直接使用 TensorRT，Forward 更易用以及更容易扩展支持更多模型和算子。目前，Forward 除了覆盖支持主流的 CV，NLP 及推荐领域的深度学习模型外，还支持一些诸如 BERT，FaceSwap，StyleTransfer 这类高级模型。

## 特点

Forward 具有以下特点：

- **模型性能优化高**：基于 TensorRT API 开发网络层级的支持，保证对于通用网络层级的推理性能优化处于最优级别。
- **模型支持范围广**：除了通用的 CV，NLP，及推荐类模型，还支持一些诸如 BERT，FaceSwap，StyleTransfer 这类高级模型。
- **多种推理模式**：支持 FLOAT / HALF / INT8 推理模式。
- **接口简单易用**：直接导入已训练好的 Tensorflow(.pb) / PyTorch(.pth) / Keras(.h5) 导出的模型文件，隐式转换为高性能的推理 Engine 进行推理加速。
- **支持自研扩展**：可根据业务模型[扩展支持自定义网络层级](doc/cn/usages/add_support_op_CN.md)。
- **支持 C++ 和 Python 接口调用**。

## 快速开始

### 环境依赖

Forward 的环境依赖如下：

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (推荐 CUDA 10.2 以上)
- TensorRT >= 6.0.1.5,  (推荐 TensorRT-7.2.1.6)
- CMake >= 3.10.1
- GCC >= 5.4.0, ld >= 2.26.1
- (Pytorch) pytorch == 1.3.1 或 pytorch == 1.7.0
- (Tensorflow) TensorFlow == 1.15.0 (linux 需额外下载【[Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases)】，将解压出来的 so 拷贝至 source/third_party/tensorflow/lib 目录下)
- (Keras) HDF 5

### 项目构建

使用 CMake 进行构建生成 Makefiles 或者 Visual Studio 项目。 根据使用目的，Forward 可构建成适用于不同框架的库, 如 Fwd-Torch, Fwd-Python-Torch, Fwd-Tf, Fwd-Python-Tf, Fwd-Keras, Fwd-Python-Keras. 构建目标由 [CMake 参数](doc/cn/usages/cmake_build_CN.md)配置, 例如, Fwd-Python-Tf 可如下配置.

``` sh
mkdir build
cd build

cmake ..  \
-DTensorRT_ROOT=/path/to/TensorRT \ 
-DENABLE_LOGGING=ON \  
-DENABLE_PROFILING=ON \  
-DENABLE_DYNAMIC_BATCH=ON \ 
-DENABLE_TORCH=OFF \  
-DENABLE_TENSORFLOW=ON \ 
-DENABLE_KERAS=OFF \ 

make -j
```

#### CMake 参数配置

- `TensorRT_ROOT` : TensorRT 安装路径
- 更多参数配置可参考 [CMake 参数](doc/cn/usages/cmake_build_CN.md)

### 单元测试

当项目构建完毕，可执行单元测试来确认项目构建是否成功。

```bash
cd build/bin
./unit_test --gtest_filter=TestTfNodes.*
```

### Forward-Python 使用

当项目构建成功后，可在 `build/bin` 目录下找到 Forward-Python 库，一般名为 `forward.cpython.xxx*.so` (Linux) 或者 `forward.xxx*.pyd` (Windows). 将该库拷贝到 Python 项目的工作目录下，例如:

```bash
---- workspace
   |
   -- test.py
   |
   -- forward.cpython.xxx*.so
```

拷贝完成后，`test.py` 可直接导入 Forward 库来进行推理加速.

```python
# test.py

import forward
import numpy as np

# 1. BUILD step: load TensorFlow-Bert model to build Forward engine
builder = forward.TfBuilder()
batch_size = 16
infer_mode = 'float32'  # Infer mode: 'float32' / 'float16' / 'int8_calib' / 'int8'

# dict_type dummy input
dummy_input = {"input_ids" : np.ones([batch_size , 48], dtype='int32'), 
               "input_mask" : np.ones([batch_size , 48], dtype='int32'),
               "segment_ids" : np.ones([batch_size , 48], dtype='int32')}

# build engine
builder.set_mode(infer_mode); # optional, 'float32' is default.
model_path = 'bert_model.pb'
tf_engine = builder.build(model_path, dummy_input)

need_save = True
if need_save:
    # save engine
    engine_path = 'path/to/out/engine'
    tf_engine.save(engine_path)

    # load saved engine
    tf_engine = forward.TfEngine()
    tf_engine.load(engine_path)

# 2. FORWARD step: do inference with Forward engine
inputs = dummy_input
outputs = tf_engine.forward(inputs) # dict_type outputs
```

**注意**: 模型输入名可通过模型查看器来查看, 例如用 [Netron](https://github.com/lutzroeder/Netron) 查看.

#### 更多使用方法

- [PyTorch 使用说明](doc/cn/usages/torch_usage_CN.md)
- [TensorFlow 使用说明](doc/cn/usages/tf_usage_CN.md)
- [Keras 使用说明](doc/cn/usages/keras_usage_CN.md)

## 常见问题

使用过程中遇到的一些问题，可先查阅 [常见问题](doc/cn/usages/FAQ_CN.md)

## 模型和算子支持

当前 Forward 的模型与算子支持如下所示，如有需要添加更多支持的，欢迎联系添加 Issue 反馈。如需要自行扩展添加支持的，可参考 [开源共建：扩展添加支持操作的流程](doc/cn/usages/add_support_op_CN.md)

### 模型

- [CV 模型](doc/operator_support.md#cv-%E6%A8%A1%E5%9E%8B)
- [语言模型](doc/operator_support.md#%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)
- [推荐模型](doc/operator_support.md#%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9E%8B)

### 算子

- [PyTorch](doc/operator_support.md#pytorch)
- [TensorFlow](doc/operator_support.md#tensorflow)

## 贡献

1. 联系进入开源共建交流讨论群。
2. 参考 [CONTRIBUTING.md](CONTRIBUTING.md) 进行开源共建。

## 贡献者

<table><tbody>
<tr><td><a target="_blank" href="https://github.com/aster2013"><img width="32px" src="https://avatars.githubusercontent.com/u/5548857?s=460&amp;v=4"> Aster JIAN </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/yuanzexi"><img width="32px" src="https://avatars.githubusercontent.com/u/14813536?s=460&amp;v=4"> Zexi YUAN </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/liao1995"><img width="32px" src="https://avatars.githubusercontent.com/u/12250510?s=460&amp;v=4"> Ao LI </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/lujq96"><img width="32px" src="https://avatars.githubusercontent.com/u/34331938?s=400&amp;v=4"> Paul LU </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/JettHu"><img width="32px" src="https://avatars.githubusercontent.com/u/35261585?s=400&amp;v=4"> JettHu </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/Ryosuke1eep"><img width="32px" src="https://avatars.githubusercontent.com/u/80682051?s=400&amp;v=4"> Ryosuke1eep </a></td></tr>
</tbody></table>

感谢所有贡献者，欢迎更多人加入一起贡献。

## 许可证

详情见 [LISENCE](LICENSE)

## 交流QQ群：776314438

![qq_group_776314438](doc/img/qq_group_776314438.png)
