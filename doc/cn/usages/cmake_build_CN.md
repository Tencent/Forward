# CMake-Build

----

## 环境依赖

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.10.1
- GCC >= 5.4.0, ld >= 2.26.1
- (Pytorch) pytorch == 1.3.1
- (Tensorflow) TensorFlow == 1.15.0 (download [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) and unzip it to `source/third_party/tensorflow/lib`)
- (Keras) HDF 5

## CMake 构建

使用 CMake 进行构建生成 Makefiles 或者 Visual Studio 项目.  根据使用目的, Forward 可构建成适用于不同框架的库, 如 Fwd-Torch, Fwd-Python-Torch, Fwd-Tf, Fwd-Python-Tf, Fwd-Keras, Fwd-Python-Keras. 构建目标由 CMake 参数配置, 例如, Fwd-Python-Tf 可如下配置.

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

### CMake 必选项参数配置

- `TensorRT_ROOT`: TensorRT 安装路径

### CMake 可选项参数配置

#### 通用配置

- `ENABLE_LOGGING`: `ON`或者`OFF`. 是否开启日志打印功能. 
- `ENABLE_PROFILING`: `ON`或者`OFF`. 是否开启性能 Profiling 功能. 
- `BUILD_PYTHON_LIB`: `ON`或者`OFF`. 是否构建 forward 的 Python 库. 若为 `ON`, 则需要同时指定 `PYTHON_EXECUTABLE`. 
- `PYTHON_EXECUTABLE`: PYTHON 可执行文件路径, 当 `BUILD_PYTHON_LIB=ON` 时, 须要指定编译时所使用的 PYTHON 路径, 须与之后的执行时用的 PYTHON 为同一个 PYTHON 来避免版本不一致的冲突. 
- `ENABLE_DYNAMIC_BATCH`: `ON`或者`OFF`. 是否开启动态 Batch 输入功能. 

#### Fwd_Torch 相关配置

- `ENABLE_TORCH` : `ON`或者`OFF`. 是否构建 Fwd_Torch 项目用于解析 PyTorch 模型. 若开启则需要设置相关的 `CMAKE_PREFIX_PATH` 或者 `PYTHON_EXECUTABLE`. 
- `CMAKE_PREFIX_PATH`: libtorch 库路径. 若 `BUILD_PYTHON_LIB=OFF` 时, 需要配置 libtorch 库路径来提供可用的 torch 库参与项目编译；若 `BUILD_PYTHON_LIB=ON` 时, 则须取消该配置, Forward 将利用 `PYTHON_EXECUTABLE` 中安装的 torch_python 库来进行编译. 

#### Fwd_Tf 相关配置

- `ENABLE_TENSORFLOW`: `ON`或者`OFF`. 是否构建 Fwd_Tf 用于解析 Tensorflow 模型. 若开启, 则需要提前根据环境依赖中的要求安装好 [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) 于 `source/third_party/tensorflow` 中. 

#### Fwd_Keras 通用配置

- `ENABLE_KERAS`: `ON`或者`OFF`. 是否构建 Fwd_Keras 用于解析 Keras 模型. 若开启, 则需要配置 `CMAKE_PREFIX_PATH`. 
- `CMAKE_PREFIX_PATH`: HDF5 库路径. 若同时需要配置 Fwd_Torch 的 `CMAKE_PREFIX_PATH`, 则可用分号相隔, 如 `CMAKE_PREFIX_PATH=/path/to/libtorch;/path/to/hdf5`. 
