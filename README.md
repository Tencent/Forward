![](doc/img/forward_logo_1.png)

# Forward - A library for high performance deep learning inference on NVIDIA GPUs

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

----

- [Forward - A library for high performance deep learning inference on NVIDIA GPUs](#forward---a-library-for-high-performance-deep-learning-inference-on-nvidia-gpus)
  - [Forward](#forward)
  - [Features](#features)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Build with CMake](#build-with-cmake)
      - [CMake build arguments](#cmake-build-arguments)
    - [Unit Test](#unit-test)
    - [Use Forward-Python](#use-forward-python)
      - [More Usages](#more-usages)
  - [FAQ](#faq)
  - [Models & Operators](#models---operators)
    - [Models](#models)
    - [Operators](#operators)
  - [Contribution](#contribution)
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
- TensorRT >= 6.0.1.5,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.10.1
- GCC >= 5.4.0, ld >= 2.26.1
- (Pytorch) pytorch == 1.3.1
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

### Use Forward-Python

When the project is successfully built, the Forward-Python library can be found in the `build/bin` directory, named as `forward.cpython.xxx*.so` in Linux or `forward.xxx*.pyd` in Windows. Forward-Python library should be copied to the workspace directory of Python project. For example, the directory is organized as:

```bash
---- workspace
   |
   -- test.py
   |
   -- forward.cpython.xxx*.so
```

Then, `test.py` can import Forward to perform high performance deep learning inference.

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

**Notice**: The name of INPUT in models can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).

#### More Usages

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

- [CV](doc/operator_support.md#cv-%E6%A8%A1%E5%9E%8B)
- [NLP](doc/operator_support.md#%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)
- [Recommender](doc/operator_support.md#%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9E%8B)

### Operators

- [PyTorch](doc/operator_support.md#pytorch)
- [TensorFlow](doc/operator_support.md#tensorflow)

## Contribution

[CONTRIBUTING](CONTRIBUTING.md)

## License

[Apache License v2.0](LICENSE)