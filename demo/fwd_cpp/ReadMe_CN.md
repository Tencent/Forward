# Demo for using Forward-Cpp in Linux

> 以 fwd_torch 为例

1. 参考 [CMake-Build](../../doc/cn/usages/cmake_build_CN.md)，构建 Forward-Cpp 库。

2. 拷贝 `build/bin` 目录下的 `libfwd_torch.so`(`libfwd_tf.so`, `libfwd_keras.so`) 和 `libtrt_engine.so` 到 `libs` 目录下.

3. 拷贝 `source/fwd_torch/torch_engine` 目录下的 `torch_engine.h`(`tf_engine.h`, `keras_engine.h`) 以及 `common` 目录下的 `common/common_macros.h` 到 `include` 目录下.

4. 修改 `build.sh` 中的 `TensorRT_ROOT` 和 `libtorch` 的地址。注意: 对于 fwd_tf 和 fwd_keras，也需要确保第三方依赖库有效，如 TensorFlow and HDF5。

5. 执行 `build.sh` 编译 demo 项目.

**注意**: 模型输入名可通过模型查看器来查看, 例如用 [Netron](https://github.com/lutzroeder/Netron) 查看.
