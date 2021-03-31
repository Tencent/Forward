![](doc/img/forward_logo_1.png)

# Forward - A library for high performance deep learning inference on NVIDIA GPUs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Build Status](https://travis-ci.com/Tencent/Forward.svg?branch=master)](https://travis-ci.com/Tencent/Forward)

----

- [Forward - A library for high performance deep learning inference on NVIDIA GPUs](#forward---a-library-for-high-performance-deep-learning-inference-on-nvidia-gpus)
  - [Forward](#forward)
  - [Features](#features)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Build with CMake](#build-with-cmake)
      - [CMake build arguments](#cmake-build-arguments)
    - [Unit Test](#unit-test)
    - [Use Forward-Cpp](#use-forward-cpp)
    - [Use Forward-Python](#use-forward-python)
    - [More Usages](#more-usages)
  - [FAQ](#faq)
  - [Models & Operators](#models---operators)
    - [Models](#models)
    - [Operators](#operators)
  - [Contribution](#contribution)
  - [Contributors](#contributors)
  - [License](#license)

----

[[中文版](README_CN.md)]

## Forward

Forward is a library for high performance deep learning inference on NVIDIA GPUs. It provides a well-designed scheme that directly parse Tensorflow/PyTorch/Keras models to high-performance engine based on [TensorRT](https://developer.nvidia.com/tensorrt). Compared to TensorRT, it is easy-to-use and easy-to-expand. So far, Forward supports not only mainstream deep learning models in CV, NLP and  Recommend fields, but also some advanced models such as BERT, GAN, FaceSwap, StyleTransfer.

## Features

- Utilize TensorRT API and customized operators for high-performance deep learning inference.
- Support not only mainstream deep learning models in CV, NLP and  Recommend fields, but also advanced models such as BERT, GAN, FaceSwap, StyleTransfer.
- Support FLOAT/HALF/INT8 infer modes.
- Easy to use: Load directly Tensorflow(.pb)/PyTorch(.pth)/Keras(.h5) models and then do inference with TensorRT.
- Easy to expand: Register customized layers refer to [add_support_op.md](doc/en/usages/add_support_op.md).
- Provide C++ and Python interfaces.

## Quick Start

### Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- (Pytorch) pytorch == 1.3.1 and pytorch == 1.7.1
- (Tensorflow) TensorFlow == 1.15.0 (download [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) and unzip it to `source/third_party/tensorflow/lib`)
- (Keras) HDF 5

### Build with CMake

Generate Makefiles or VS project (Windows) and build. Forward can be built for different framework, such as Fwd-Torch, Fwd-Python-Torch, Fwd-Tf, Fwd-Python-Tf, Fwd-Keras, Fwd-Python-Keras, which controlled by [CMake options](doc/en/usages/cmake_build.md). For example, Fwd-Python-Tf is built as below.

``` sh
mkdir build
cd build

cmake ..  \
-DTensorRT_ROOT=/path/to/TensorRT \ 
-DENABLE_LOGGING=ON \  
-DENABLE_PROFILING=ON \  
-DENABLE_DYNAMIC_BATCH=ON \ 
-DBUILD_PTYHON_LIB=ON \
-DENABLE_TORCH=OFF \  
-DENABLE_TENSORFLOW=ON \ 
-DENABLE_KERAS=OFF \ 

make -j
```

#### CMake build arguments

- `TensorRT_ROOT` [Required]: Path to the TensorRT installation directory containing libraries
- More CMake options refer to [CMake Options](doc/en/usages/cmake_build.md)

### Unit Test

When the project is built, unit_test can be used to verify the project is successfully built.

```bash
cd build/bin
./unit_test --gtest_filter=TestTfNodes.*
```

### Use Forward-Cpp

Refer to [Demo for using Forward-Cpp in Linux](demo/fwd_cpp/ReadMe.md)

### Use Forward-Python

Refer to [Demo for using Forward-Python](demo/fwd_py/ReadMe.md)

### More Usages

**Notice**: The name of INPUT in models can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).

- [PyTorch usages](doc/en/usages/torch_usage.md)
- [TensorFlow usages](doc/en/usages/tf_usage.md)
- [Keras usages](doc/en/usages/keras_usage.md)

## FAQ

[FAQ](doc/en/usages/FAQ.md)

- [Infer modes](doc/en/usages/#Infer-modes)
- [engine with INT8 mode cannot be built](doc/en/usages/#engine-with-INT8-mode-cannot-be-built)
- [Core dumped in multi-thread scenarios](doc/en/usages/#Core-dumped-in-multi-thread-scenarios)

## Models & Operators

### Models

- [CV](doc/operator_support.md#cv)
- [NLP](doc/operator_support.md#nlp)
- [Recommender](doc/operator_support.md#recommender)

### Operators

- [PyTorch](doc/operator_support.md#pytorch)
- [TensorFlow](doc/operator_support.md#tensorflow)

## Contribution

[CONTRIBUTING](CONTRIBUTING.md)

## Contributors

<table><tbody>
<tr><td><a target="_blank" href="https://github.com/aster2013"><img width="32px" src="https://avatars.githubusercontent.com/u/5548857?s=460&amp;v=4"> Aster JIAN </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/yuanzexi"><img width="32px" src="https://avatars.githubusercontent.com/u/14813536?s=460&amp;v=4"> Zexi YUAN </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/liao1995"><img width="32px" src="https://avatars.githubusercontent.com/u/12250510?s=460&amp;v=4"> Ao LI </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/lujq96"><img width="32px" src="https://avatars.githubusercontent.com/u/34331938?s=400&amp;v=4"> Paul LU </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/JettHu"><img width="32px" src="https://avatars.githubusercontent.com/u/35261585?s=400&amp;v=4"> JettHu </a></td></tr>
<tr><td><a target="_blank" href="https://github.com/Ryosuke1eep"><img width="32px" src="https://avatars.githubusercontent.com/u/80682051?s=400&amp;v=4"> Ryosuke1eep </a></td></tr>
</tbody></table>

Any form of contribution is welcome. The above contributors have been officially released by Tencent.

We very much welcome developers to contribute to Tencent's open source, and we will also give them incentives to acknowledge and thank them. Here we provide an official description of Tencent's open source contribution. Specific contribution rules for each project are formulated by the project team. Developers can choose the appropriate project and participate according to the corresponding rules. The Tencent Project Management Committee will report regularly to qualified contributors and awards will be issued by the official contact.

## License

[Apache License v2.0](LICENSE)