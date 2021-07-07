![](doc/img/forward_logo_1.png)

# Forward 深度学习推理加速框架

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Build Status](https://travis-ci.com/Tencent/Forward.svg?branch=master)](https://travis-ci.com/Tencent/Forward)

----

- [Forward 深度学习推理加速框架](#forward-深度学习推理加速框架)
  - [什么是 Forward](#什么是-forward)
  - [为什么选择 Forward](#为什么选择-forward)
  - [快速上手 Forward](#快速上手-forward)
    - [环境依赖](#环境依赖)
    - [项目构建](#项目构建)
    - [Forward-Cpp 使用](#forward-cpp-使用)
    - [Forward-Python 使用](#forward-python-使用)
    - [Forward-Bert 使用](#forward-bert-使用)
    - [更多使用方法](#更多使用方法)
    - [Logging 日志](#logging-日志)
  - [模型和算子支持](#模型和算子支持)
    - [模型](#模型)
    - [算子](#算子)
  - [参考资料](#参考资料)
  - [贡献](#贡献)
  - [许可证](#许可证)

----

[[English Version](README_EN.md)]

## 什么是 Forward

Forward 是一款腾讯研发的 GPU 高性能推理加速框架。它提出了一种解析方案，可直接加载主流框架模型（Tensorflow / PyTorch / Keras）转换成 TensorRT 推理加速引擎，帮助用户节省中间繁杂的模型转换或网络构建步骤。相对于直接使用 TensorRT，Forward 更易用以及更容易扩展支持更多模型和算子。目前，Forward 除了覆盖支持主流的 CV，NLP 及推荐领域的深度学习模型外，还支持一些诸如 BERT，FaceSwap，StyleTransfer 这类高级模型。

## 为什么选择 Forward

- **模型性能优化高**：基于 TensorRT API 开发网络层级的支持，保证对于通用网络层级的推理性能优化处于最优级别。
- **模型支持范围广**：除了通用的 CV，NLP，及推荐类模型，还支持一些诸如 BERT，FaceSwap，StyleTransfer 这类高级模型。
- **多种推理模式**：支持 FLOAT / HALF / INT8 推理模式。
- **接口简单易用**：直接导入已训练好的 Tensorflow(.pb) / PyTorch(.pth) / Keras(.h5) 导出的模型文件，隐式转换为高性能的推理 Engine 进行推理加速。
- **支持自研扩展**：可根据业务模型[扩展支持自定义网络层级](doc/cn/usages/add_support_op_CN.md)。
- **支持 C++ 和 Python 接口调用**。

## 快速上手 Forward

### 环境依赖

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (推荐 CUDA 10.2 以上)
- TensorRT >= 7.0.0.11 (推荐 TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- PyTorch >= 1.7.0
- TensorFlow >= 1.15.0 (若使用 Linux 操作系统，需额外下载 [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases)，并将解压出来的 `.so` 文件拷贝至 `Forward/source/third_party/tensorflow/lib` 目录下)
- Keras HDF5 (从 `Forward/source/third_party/hdf5` 源码构建)

### 项目构建

使用 CMake 进行构建生成 Makefiles 或者 Visual Studio 项目。根据使用目的，Forward 可构建成适用于不同框架的库，如 Fwd-Torch、 Fwd-Python-Torch、 Fwd-Tf、 Fwd-Python-Tf、 Fwd-Keras 和 Fwd-Python-Keras。

以 Linux 平台构建 Fwd-Tf 为例，

步骤一：克隆项目
```bash
1 git clone https://github.com/Tencent/Forward.git
```
步骤二：下载 `Tensorflow 1.15.0`（仅在 Linux 平台使用 Tensorflow 框架推理时需要）
```bash
1 cd Forward/source/third_party/tensorflow/
2 wget https://github.com/neargye-forks/tensorflow/releases/download/v1.15.0/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
3 tar -xvf libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz -C lib/
```
步骤三：创建 `build` 文件夹
```bash
1 cd ~/Forward/
2 rm -rf build
3 mkdir -p build
4 cd build/
```
步骤四：使用 `cmake` 生成构建关系，需指定 `TensorRT_ROOT` 安装路径
```bash
1 cmake ..  -DTensorRT_ROOT=<path_to_TensorRT> -DENABLE_TENSORFLOW=ON -DENABLE_UNIT_TESTS=ON
```
步骤五：使用 `make` 构建项目
```bash
1 make -j
```
步骤六：运行 `unit_test` 验证项目是否构建成功
```bash
cd bin/
./unit_test --gtest_filter=TestTfNodes.*

# 出现已下提示表示项目构建成
# [       OK ] TestTfNodes.ZeroPadding (347 ms)
# [----------] 22 tests from TestTfNodes (17555 ms total)

# [----------] Global test environment tear-down
# [==========] 22 tests from 1 test case ran. (17555 ms total)
# [  PASSED  ] 22 tests.
```

更多构建流程可参考 [CMake 构建流程](doc/cn/usages/cmake_build_CN.md) 。

### Forward-Cpp 使用

参考 [Demo for using Forward-Cpp in Linux](demo/fwd_cpp/ReadMe_CN.md)

### Forward-Python 使用

参考 [Demo for using Forward-Python](demo/fwd_py/ReadMe_CN.md)

### Forward-Bert 使用

Refer to [Demo for using Forward-Bert](demo/bert/README_CN.md)

### 更多使用方法

**注意**: 模型输入名可通过模型查看器来查看, 例如用 [Netron](https://github.com/lutzroeder/Netron) 查看。

- [PyTorch 使用说明](doc/cn/usages/torch_usage_CN.md)
- [TensorFlow 使用说明](doc/cn/usages/tf_usage_CN.md)
- [Keras 使用说明](doc/cn/usages/keras_usage_CN.md)

### Logging 日志

Forward 使用 [easylogging++](https://github.com/amrayn/easyloggingpp) 作为日志功能，并使用 `forward_log.conf` 作为日志配置文件。 

- 若工作目录中存在 `forward_log.conf` 文件，Forward 将使用该配置文件，更多内容可参考 [Using-configuration-file](https://github.com/amrayn/easyloggingpp#using-configuration-file) 。
- 若工作目录中不存在 `forward_log.conf` 文件，Forward 将使用默认配置，并将日志记录到 `logs/myeasylog.log` 。

`forward_log.conf` 文件配置样例
```bash
* GLOBAL:
  FORMAT               =  "[%level] %datetime %fbase(%line): %msg"
  FILENAME             =  "Forward.log"
  ENABLED              =  true
  TO_FILE              =  true
  TO_STANDARD_OUTPUT   =  true
  PERFORMANCE_TRACKING =  true
  MAX_LOG_FILE_SIZE    =  2097152 ## 2MB - Comment starts with two hashes (##)
  LOG_FLUSH_THRESHOLD  =  100 ## Flush after every 100 logs
```

## 模型和算子支持

当前 Forward 的模型与算子支持如下所示，如有需要添加更多支持的，欢迎联系添加 Issue 反馈。如需要自行扩展添加支持的，可参考 [开源共建：扩展添加支持操作的流程](doc/cn/usages/add_support_op_CN.md)

### 模型

- [CV 模型](doc/operator_support.md#cv)
- [语言模型](doc/operator_support.md#nlp)
- [推荐模型](doc/operator_support.md#recommender)

### 算子

- [PyTorch](doc/operator_support.md#pytorch)
- [TensorFlow](doc/operator_support.md#tensorflow)

## 参考资料
1. [推理流程构建过程](doc/cn/usages/forward_workflow_CN.md)
2. [推理引擎使用方法](doc/cn/usages/inference_engine_usage_CN.md)
3. [工具与测试](doc/cn/usages/tool_and_test_CN.md)
4. [常见问题](doc/cn/usages/FAQ_CN.md)

## 贡献   

1. 联系进入开源共建交流讨论群，QQ 群：[776314438](doc/img/qq_group_776314438.png)。
2. 请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 进行开源共建。
   
<table>
<tr>
   <td align="center"><a href="https://github.com/aster2013"><img src="https://avatars.githubusercontent.com/u/5548857?s=460&amp;v=4" width="32px;" alt=""/><br /><sub><b>Aster JIAN</b></sub></a></td>
   <td align="center"><a href="https://github.com/yuanzexi"><img src="https://avatars.githubusercontent.com/u/14813536?s=460&amp;v=4" width="32px;" alt=""/><br /><sub><b>Zexi YUAN</b></sub></a></td>
   <td align="center"><a href="https://github.com/liao1995"><img src="https://avatars.githubusercontent.com/u/12250510?s=460&amp;v=4" width="32px;" alt=""/><br /><sub><b>Ao LI</b></sub></a></td>
   <td align="center"><a href="https://github.com/lujq96"><img src="https://avatars.githubusercontent.com/u/34331938?s=400&amp;v=4" width="32px;" alt=""/><br /><sub><b>Paul LU</b></sub></a></td>
   <td align="center"><a href="https://github.com/zhaoyiluo"><img src="https://avatars.githubusercontent.com/u/14813536?s=460&amp;v=4" width="32px;" alt=""/><br /><sub><b>Zhaoyi LUO</b></sub></a></td>
   <td align="center"><a href="https://github.com/JettHu"><img src="https://avatars.githubusercontent.com/u/35261585?s=400&amp;v=4" width="32px;" alt=""/><br /><sub><b>Jett Hu</b></sub></a></td>
   <td align="center"><a href="https://github.com/Ryosuke1eep"><img src="https://avatars.githubusercontent.com/u/80682051?s=400&amp;v=4" width="32px;" alt=""/><br /><sub><b>Ryosuke1eep</b></sub></a></td>
</tr>
</table>

感谢所有贡献者，欢迎更多人加入一起贡献。

## 许可证

详情见 [LISENCE](LICENSE)
