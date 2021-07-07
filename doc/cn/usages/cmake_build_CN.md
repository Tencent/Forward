# CMake-Build

## 环境依赖

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (推荐 CUDA 10.2 以上)
- TensorRT >= 7.0.0.11 (推荐 TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- PyTorch >= 1.7.0
- TensorFlow >= 1.15.0 (若使用 Linux 操作系统，需额外下载 [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases)，并将解压出来的 .so 文件拷贝至 `Forward/source/third_party/tensorflow/lib` 目录下)
- Keras HDF5 (从 `Forward/source/third_party/hdf5` 源码构建)

## CMake 构建流程

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

## CMake 构建参数

<table>
    <tr>
        <td nowrap="nowrap">配置</td> 
        <td nowrap="nowrap">参数名</td>
        <td nowrap="nowrap">可选值</td>
        <td nowrap="nowrap">默认值</td>
        <td>功能</td>
        <td>备注</td>
   </tr>
   <tr>
        <td rowspan="9" nowrap="nowrap">通用</td>
        <td>TensorRT_ROOT</td>
        <td>path_to_TensorRT</td>
        <td>N/A</td>
        <td>指定 TensorRT 安装路径</td>
        <td>必选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_LOGGING</td>
        <td>N/A</td>
        <td><code>ON</code></td>
        <td>开启日志打印功能；<br>可通过修改 <code>forward_log.conf</code> 配置文件关闭该功能</td>
        <td>N/A</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_PROFILING</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否开启性能 Profiling 功能</td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">BUILD_PYTHON_LIB</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否构建 Forward 的 <code>Python</code> 库；<br>若为 <code>ON</code>，需另配置 <code>PYTHON_EXECUTABLE</code> 参数</td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">PYTHON_EXECUTABLE</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>指定 <code>Python</code> 可执行文件路径，需与实际使用时的 <code>Python</code> 相同，避免版本不一致产生的冲突</td>
        <td>配合 <code>BUILD_PYTHON_LIB</code> 参数</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_DYNAMIC_BATCH</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否开启动态 Batch 输入功能</td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_RNN</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否开启 RNN 模型推理</td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_INFER_TESTS</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否开启推理测试；<br>若为 <code>ON</code>，需另配置 <code>OpenCV</code> 库</td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_UNIT_TEST</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否开启单元测试</td>
        <td>可选</td>
   </tr>
   <tr>
        <td rowspan="3" nowrap="nowrap">PyTorch</td>
        <td>ENABLE_TORCH</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否构建用于解析 PyTorch 模型的 Fwd_Torch 项目；<br>若为 <code>ON</code>，需另配置 <code>CMAKE_PREFIX_PATH</code> 或 <code>PYTHON_EXECUABLE</code> 参数</td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">ENABLE_TORCH_PLUGIN</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否开启 Torch 子模块插件，该插件能支持子模块中更多的 Torch-Op，但不保证性能的提升；<br>若为 <code>ON</code>，<code>libtrt_engine.so</code> 需依赖 <code>Torch-Libraries</<code>code></td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">CMAKE_PREFIX_PATH</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>指定 <code>LibTorch</code> 库路径；<br>若 <code>BUILD_PYTHON_LIB=OFF</code> 时，需另配置 LibTorch 库进行编译；<br>若 <code>BUILD_PYTHON_LIB=ON</code> 时，则需取消该配置，Forward 将利用 <code>PYTHON_EXECUTABLE</code> 中安装的 <code>torch_python</code> 库进行编译</td>
        <td>配合 <code>ENABLE_TORCH</code> 或 <code>ENABLE_KERAS</code> 参数</td>
   </tr>
   <tr>
        <td rowspan="1" nowrap="nowrap">TensorFlow</td>
        <td>ENABLE_TENSORFLOW</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否构建用于解析 TensorFlow 模型的 Fwd_Tf 项目；<br>若为 <code>ON</code>，需提前根据环境依赖中的要求在 <code>source/third_party/tensorflow/bin</code> 目录下安装 <code>Tensorflow 1.15.0</code></td>
        <td>可选</td>
   </tr>
   <tr>
        <td rowspan="2" nowrap="nowrap">Keras</td>
        <td>ENABLE_KERAS</td>
        <td><code>ON</code> 或 <code>OFF</code></td>
        <td><code>OFF</code></td>
        <td>是否构建用于解析 Keras 模型的 Fwd_Keras 项目；<br>若为 <code>ON</code>，需同时配置 <code>CMAKE_PREFIX_PATH</code></td>
        <td>可选</td>
   </tr>
   <tr>
        <td nowrap="nowrap">CMAKE_PREFIX_PATH</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>指定 <code>HDF5</code> 库路径；<br>若同时配置 Fwd_Torch 的 <code>CMAKE_PREFIX_PATH</code>，可用分号隔开，如 <code>/path/to/libtorch;/path/to/hdf5</code></td>
        <td>配合 <code>ENABLE_TORCH</code> 或 <code>ENABLE_KERAS</code> 参数</td>
   </tr>
</table>