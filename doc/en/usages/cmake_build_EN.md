# CMake-Build

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: >= CUDA 10.2)
- TensorRT >= 7.0.0.11 (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- PyTorch >= 1.7.0
- TensorFlow >= 1.15.0 (For Linux users, download [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) and unzip all `.so` files under `Forward/source/third_party/tensorflow/lib` folder)
- Keras HDF5 (Built from `Forward/source/third_party/hdf5` as default)

## Build with CMake

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

## CMake Build Parameters

<table>
    <tr>
        <td nowrap="nowrap">Config</td> 
        <td nowrap="nowrap">Parameter</td>
        <td nowrap="nowrap">Optional<br>Value</td>
        <td nowrap="nowrap">Default<br>Value</td>
        <td>Content</td>
        <td>Remark</td>
   </tr>
   <tr>
        <td rowspan="8" nowrap="nowrap">General</td>
        <td nowrap="nowrap">TensorRT_ROOT</td>
        <td>path_to_TensorRT</td>
        <td>N/A</td>
        <td>Specify TensorRT installation path</td>
        <td>Required</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_PROFILING</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Enable Profiling</td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">BUILD_PYTHON_LIB</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Build Forward library in <code>Python</code>;<br>If <code>ON</code>, should also configure <code>PYTHON_EXECUTABLE</code></td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">PYTHON_EXECUTABLE</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>Specify <code>Python</code> executable path, should be the same path when using <code>Python</code> in the workspace environment to avoid conflicts caused by inconsistent versions</td>
        <td>Use with <code>BUILD_PYTHON_LIB</code></td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_DYNAMIC_BATCH</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Enable dynamic batch input</td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_RNN</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Enable RNN model inference</td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_INFER_TESTS</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Enable inference tests;<br>If <code>ON</code>, should also configure <code>OpenCV</code> library</td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_UNIT_TEST</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Enable unit tests</td>
        <td>Fwd_Keras unit test depends on Fwd_Tf；<br>Fwd_Onnx unit test depends on Fwd_Torch</td>
   </tr>
   <tr>
        <td rowspan="3" nowrap="nowrap">PyTorch</td>
        <td nowrap="nowrap">ENABLE_TORCH</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Build Fwd_Torch to parse PyTorch models;<br>If <code>ON</code>, should also configure <code>CMAKE_PREFIX_PATH</code> or <code>PYTHON_EXECUABLE</code></td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_TORCH_PLUGIN</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Enable Torch submodule plugin to support more Torch-Op with TorchSubmodule, but the performance is not guaranteed to increase<br>If <code>ON</code>, <code>libtrt_engine.so</code> should use <code>Torch-Libraries</<code>code></td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">CMAKE_PREFIX_PATH</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>Specify <code>LibTorch</code> library path;<br>If <code>BUILD_PYTHON_LIB=OFF</code>, should also configure LibTorch library during compiling step;<br>If <code>BUILD_PYTHON_LIB=ON</code>, Forward will compile with <code>torch_python</code> specified by <code>PYTHON_EXECUTABLE</code> and cancel this setting</td>
        <td>Use with <code>ENABLE_TORCH</code> or <code>ENABLE_KERAS</code></td>
   </tr>
   <tr>
        <td rowspan="1" nowrap="nowrap">TensorFlow</td>
        <td nowrap="nowrap">ENABLE_TENSORFLOW</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Build Fwd_Tf to parse TensorFlow models;<br>If <code>ON</code>, should also install <code>Tensorflow 1.15.0</code> under <code>source/third_party/tensorflow/bin</code> folder</td>
        <td>Optional</td>
   </tr>
   <tr>
        <td rowspan="2" nowrap="nowrap">Keras</td>
        <td nowrap="nowrap">ENABLE_KERAS</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Build Fwd_Keras to parse Keras models;<br>If <code>ON</code>, should also configure <code>CMAKE_PREFIX_PATH</code></td>
        <td>Optional</td>
   </tr>
   <tr>
        <td nowrap="nowrap">CMAKE_PREFIX_PATH</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>Specify <code>HDF5</code> library path;<br>If Fwd_Torch is built at the same time, <code>CMAKE_PREFIX_PATH</code> can be separated by semicolons, for example, <code>/path/to/libtorch;/path/to/hdf5</code></td>
        <td>Use with <code>ENABLE_TORCH</code> or <code>ENABLE_KERAS</code></td>
   </tr>
   <tr>
        <td rowspan="1" nowrap="nowrap">ONNX</td>
        <td>ENABLE_ONNX</td>
        <td><code>ON</code> or <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>Build Fwd_Onnx to parse ONNX models</td>
        <td>Optional</td>
   </tr>
</table>
