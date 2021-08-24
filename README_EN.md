![](doc/img/forward_logo_1.png)

# Forward - A library for high performance deep learning inference on NVIDIA GPUs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Build Status](https://travis-ci.com/Tencent/Forward.svg?branch=master)](https://travis-ci.com/Tencent/Forward)

----

- [Forward - A library for high performance deep learning inference on NVIDIA GPUs](#forward---a-library-for-high-performance-deep-learning-inference-on-nvidia-gpus)
  - [What is Forward](#what-is-forward)
  - [Why choose Forward](#why-choose-forward)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Build with CMake](#build-with-cmake)
    - [Forward-Cpp Usage](#forward-cpp-usage)
    - [Forward-Python Usage](#forward-python-usage)
    - [Forward-Bert Usage](#forward-bert-usage)
    - [More Usages](#more-usages)
    - [Logging](#logging)
  - [Models & Operators](#models--operators)
    - [Models](#models)
    - [Operators](#operators)
  - [References](#references)
  - [Contributing](#contributing)
  - [License](#license)

----

[[中文版](README_CN.md)]

## What is Forward

Forward is a library for high-performance deep learning inference on NVIDIA GPUs. It provides a well-designed scheme that directly parses Tensorflow/PyTorch/Keras/ONNX models to the high-performance engine based on [TensorRT](https://developer.nvidia.com/tensorrt). Compared to TensorRT, it is easy-to-use and easy-to-expand. So far, Forward supports not only mainstream deep learning models in CV, NLP, and  Recommend fields, but also some advanced models such as BERT, GAN, FaceSwap, StyleTransfer.

## Why choose Forward

- **High-performance optimization**: utilize TensorRT API and customized operators for high-performance deep learning inference;
- **Wide range support**: support advanced models such as BERT, GAN, FaceSwap, and StyleTransfer besides the mainstream deep learning models in CV, NLP, and Recommend fields;
- **Multiple modes**: support FLOAT/HALF/INT8 infer modes;
- **Easy to use**: load Tensorflow(.pb)/PyTorch(.pth)/Keras(.h5)/ONNX(.onnx) models directly, followed by inferencing with TensorRT;
- **Easy to expand**: register customized layers according to [add_support_op.md](doc/en/usages/add_support_op_EN.md);
- **Provide C++ and Python interfaces**.

## Quick Start

### Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: >= CUDA 10.2)
- TensorRT >= 7.0.0.11 (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- PyTorch >= 1.7.0
- TensorFlow >= 1.15.0 (For Linux users, download [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) and unzip all `.so` files under `Forward/source/third_party/tensorflow/lib` folder)
- Keras HDF5 (Built from `Forward/source/third_party/hdf5` as default)

### Build with CMake

Use CMake to generate Makefiles or Visual Studio project (Windows). Upon the usage, Forward is able to be built for different frameworks, such as Fwd-Torch, Fwd-Python-Torch, Fwd-Tf, Fwd-Python-Tf, Fwd-Keras, Fwd-Python-Keras, Fwd-Onnx, and Fwd-Python-Onnx.

The example below shows the building workflow under the Linux system,

Step 1: clone the project
```bash
1 git clone https://github.com/Tencent/Forward.git
```
Step 2: download `Tensorflow 1.15.0` (only needed when using Forward with Tensorflow framework under Linux)
```bash
1 cd Forward/source/third_party/tensorflow/
2 wget https://github.com/neargye-forks/tensorflow/releases/download/v1.15.0/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
3 tar -xvf libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
```
Step 3: create `build` folder
```bash
1 cd ~/Forward/
2 rm -rf build
3 mkdir -p build
4 cd build/
```
Step 4: run `cmake` to create dependencies and `TensorRT_ROOT` installation path should be specified
```bash
1 cmake ..  -DTensorRT_ROOT=<path_to_TensorRT> -DENABLE_TENSORFLOW=ON -DENABLE_UNIT_TESTS=ON
```
Step 5: run `make` to build the project
```bash
1 make -j
```
Step 6: run `unit_test` to check if the project has been built built successfully
```bash
cd bin/
./unit_test --gtest_filter=TestTfNodes.*

# The following lines show the success of building the project 
# [       OK ] TestTfNodes.ZeroPadding (347 ms)
# [----------] 22 tests from TestTfNodes (17555 ms total)

# [----------] Global test environment tear-down
# [==========] 22 tests from 1 test case ran. (17555 ms total)
# [  PASSED  ] 22 tests.
```

For more information, please refer to [CMake Build](doc/en/usages/cmake_build_EN.md).

### Forward-Cpp Usage

Refer to [Demo for using Forward-Cpp in Linux](demo/fwd_cpp/ReadMe.md)

### Forward-Python Usage

Refer to [Demo for using Forward-Python](demo/fwd_py/ReadMe.md)

### Forward-Bert Usage

Refer to [Demo for using Forward-Bert](demo/bert/README.md)

### More Usages

**Notice**: the model INPUT name can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).

- [PyTorch usages](doc/en/usages/torch_usage_EN.md)
- [TensorFlow usages](doc/en/usages/tf_usage_EN.md)
- [Keras usages](doc/en/usages/keras_usage_EN.md)
- [ONNX usages](doc/en/usages/onnx_usage_EN.md)

### Logging

Forward use [easylogging++](https://github.com/amrayn/easyloggingpp) for logging, and use `forward_log.conf` as the log configuration file. 

- If `forward_log.conf` is existed under the workspace directory, Forward will use this file as the configuration file. For more information, please refer to [Using-configuration-file](https://github.com/amrayn/easyloggingpp#using-configuration-file);
- If `forward_log.conf` is not existed under the workspace directory, Forward will use its default settings and save logging information in `logs/myeasylog.log`.

Example for the log configuration file `forward_log.conf`
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

## Models & Operators

All the models and operators supported by Forward are listed below. If the one you are looking for is not listed here, please create a new Issue. We also welcome developers to contribute to this project together. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

### Models

- [CV](doc/operator_support.md#cv)
- [NLP](doc/operator_support.md#nlp)
- [Recommender](doc/operator_support.md#recommender)

### Operators

- [PyTorch](doc/operator_support.md#pytorch)
- [TensorFlow](doc/operator_support.md#tensorflow)

## References

1. [Forward Workflow](doc/en/usages/forward_workflow_EN.md)
2. [Inference Engine Usage](doc/en/usages/inference_engine_usage_EN.md)
3. [Tool and Test](doc/en/usages/tool_and_test_EN.md)
4. [FAQ](doc/en/usages/FAQ_EN.md)

## Contributing

1. Welcome to join our discussion group, Tencent QQ Group ID: [776314438](doc/img/qq_group_776314438.png).
2. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) about how to contribute to this project.
   
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

Any form of contribution is welcome. The above contributors have been officially released by Tencent.

We very much welcome developers to contribute to Tencent's open source, and we will also give them incentives to acknowledge and thank them. Here we provide an official description of Tencent's open source contribution. Specific contribution rules for each project are formulated by the project team. Developers can choose the appropriate project and participate according to the corresponding rules. The Tencent Project Management Committee will report regularly to qualified contributors and awards will be issued by the official contact.

## License

[Apache License v2.0](LICENSE)