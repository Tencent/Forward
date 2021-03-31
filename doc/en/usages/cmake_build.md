# CMake-Build

----

- [CMake-Build](#cmake-build)
  - [Prerequisites](#prerequisites)
  - [Build with CMake](#build-with-cmake)
    - [Required CMake build arguments](#required-cmake-build-arguments)
    - [Optional CMake build arguments](#optional-cmake-build-arguments)
      - [Common](#common)
      - [Fwd_Torch related](#fwd-torch-related)
      - [Fwd_Tf related](#fwd-tf-related)
      - [Fwd_Keras related](#fwd-keras-related)


----

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- (Pytorch) pytorch == 1.3.1
- (Tensorflow) TensorFlow == 1.15.0 (download [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) and unzip it to `source/third_party/tensorflow/lib`)
- (Keras) HDF 5

## Build with CMake

Generate Makefiles or VS project (Windows) and build. Forward can be built for different framework, such as Fwd-Torch, Fwd-Python-Torch, Fwd-Tf, Fwd-Python-Tf, Fwd-Keras, Fwd-Python-Keras, which controlled by CMake options. For example, Fwd-Python-Tf is built as below. 

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

### Required CMake build arguments

- `TensorRT_ROOT`: Path to the TensorRT installation directory containing libraries

### Optional CMake build arguments

#### Common

- `ENABLE_LOGGING`: Specifying if logging is enabled, for example [`ON`] | `OFF`.
- `ENABLE_PROFILING`: Specifying if profiling is enabled, for example [`ON`] | `OFF`.
- `BUILD_PYTHON_LIB`: Specifying if Forward Python SDK should be built, for example [`ON`] | `OFF`. If `BUILD_PYTHON_LIB=ON`, `PYTHON_EXECUTABLE` should also be specified.
- `PYTHON_EXECUTABLE`: Path to executable Python binary file for building Forward. It should be specified when `BUILD_PYTHON_LIB=ON`. Note: The Python used for building Forward library should be the same as the Python used for importing the Forward library.
- `ENABLE_DYNAMIC_BATCH`: Specifying if dynamic batch input is enabled, for example [`ON`] | `OFF`.

#### Fwd_Torch related

- `ENABLE_TORCH` : Specifying if PyTorch-related parsers should be built, for example [`ON`] | `OFF`.  If `ENABLE_TORCH=ON` and `BUILD_PYTHON_LIB=OFF`, `CMAKE_PREFIX_PATH` should be specified. If `ENABLE_TORCH=ON` and `BUILD_PYTHON_LIB=ON`, `PYTHON_EXECUTABLE` should be specified.
- `CMAKE_PREFIX_PATH`: Path to the libtorch installation directory containing libraries. If `BUILD_PYTHON_LIB=OFF`, it should be specified for building PyTorch-related parsers. If `BUILD_PYTHON_LIB=ON`, it should be not specified and `PYTHON_EXECUTABLE` should be specified.

#### Fwd_Tf related

- `ENABLE_TENSORFLOW`: Specifying if Tensorflow-related parsers should be built, for example [`ON`] | `OFF`. If `ENABLE_TENSORFLOW=ON`, [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) libraries should be installed in `source/third_party/tensorflow`.

#### Fwd_Keras related

- `ENABLE_KERAS`: Specifying if Keras-related parsers should be built, for example [`ON`] | `OFF`. If `ENABLE_KERAS=ON`, `CMAKE_PREFIX_PATH` should be specified.
- `CMAKE_PREFIX_PATH`: Path to HDF5 libraries. If `ENABLE_TORCH=ON` and `CMAKE_PREFIX_PATH` should be specified for PyTorch-related parsers, the path to HDF5 libraries can be append to `CMAKE_PREFIX_PATH` by comma. For example, `CMAKE_PREFIX_PATH=/path/to/libtorch;/path/to/hdf5`.
