# Demo for using Forward-Cpp in Linux

> Use fwd_torch as example

1. Build Forward-Cpp libraries refer to [CMake-Build](../../doc/en/usages/cmake_build.md).

2. Copy `libfwd_torch.so`(`libfwd_tf.so`, `libfwd_keras.so`) and `libtrt_engine.so` in the `build/bin` into the `libs` directory.

3. Copy `torch_engine.h`(`tf_engine.h`, `keras_engine.h`) and `common/common_macros.h` in the `source/fwd_torch/torch_engine` into the `include` directory.

4. Change the path of `TensorRT_ROOT` and `libtorch` in the script `build.sh`. Notice: for fwd_tf and fwd_keras, you should make sure the third dependencies are valid, such as TensorFlow and HDF5.

5. run the script `build.sh` to build the demo project.

**Notice**: The name of INPUT in models can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).
