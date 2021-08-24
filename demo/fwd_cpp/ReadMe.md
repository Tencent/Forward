# Demo for using Forward-Cpp in Linux

1. Build Forward-Cpp libraries refer to [CMake-Build](../../doc/en/usages/cmake_build_EN.md).

2. Copy `libfwd_torch.so` (`libfwd_tf.so`, `libfwd_keras.so`, `libfwd_onnx.so`) and `libtrt_engine.so` from the `build/bin` to the `libs` directory.

3. Change the path of `TensorRT_ROOT` in the script `build.sh`. If you need test `test_fwd_torch.cpp` or/and `test_fwd_tf.cpp` or/and `test_fwd_onnx.cpp`, then you should specified `ENABLE_TORCH=ON` or/and `ENABLE_TENSORFLOW=ON` or/and `ENABLE_ONNX=ON`, and `LibTorch` or/and `LibTensorflow`.

4. Update the `model_path` and `input informations` in the `test_fwd_engine.cpp`, `test_fwd_tf.cpp` or/and `test_fwd_torch.cpp` or/and `test_fwd_onnx.cpp`.

4. run the script `build.sh` to build the demo project.

**Notice**: The name of INPUT in models can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).
