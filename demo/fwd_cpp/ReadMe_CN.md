# Demo for using Forward-Cpp in Linux

1. 参考 [CMake-Build](../../doc/cn/usages/cmake_build_CN.md)，构建 Forward-Cpp 库。

2. 拷贝 `build/bin` 目录下的 `libfwd_torch.so` (`libfwd_tf.so`，`libfwd_keras.so`，`libfwd_onnx.so`) 和 `libtrt_engine.so` 到 `libs` 目录下。

3. 修改 `build.sh` 中的 `TensorRT_ROOT`。如果需要测试 `test_fwd_torch.cpp` 或者 `test_fwd_tf.cpp` 或者 `test_fwd_onnx.cpp`，则需要同时修改 `ENABLE_TORCH=ON` 或 `ENABLE_TENSORFLOW=ON` 或 `ENABLE_ONNX=ON`，以及 `LibTorch` 或 `LibTensorflow` 。

4. 更新 `test_fwd_engine.cpp`, `test_fwd_tf.cpp` 或 `test_fwd_torch.cpp` 或 `test_fwd_onnx.cpp` 中的 `model_path` and `input informations` 。

5. 执行 `build.sh` 编译 demo 项目。

**注意**: 模型输入名可通过模型查看器来查看, 例如用 [Netron](https://github.com/lutzroeder/Netron) 查看。
