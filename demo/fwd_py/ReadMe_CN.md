# Demo for using Forward-Python

1. 参考 [CMake-Build](../../doc/cn/usages/cmake_build_CN.md)，构建 Forward-Python 库。

2. 拷贝 `build/bin` 目录下的 `forward.cpython.xxx*.so` (Linux) or `forward.xxx*.pyd` (Windows) 到 Python 的工作目录下。则目录组织结构如下:

```bash
---- workspace
   |
   -- test_forward_torch.py
   |
   -- test_forward_tf.py
   |
   -- test_forward_onnx.py
   |
   -- softmax.pb
   |
   -- softmax.pt
   |
   -- softmax.onnx
   |
   -- forward.cpython.xxx*.so
```

3. 使用构建 (CMake-Build) 时使用的 Python 来运行并导入 Forward。（`import forward`）

**注意**: 模型输入名可通过模型查看器来查看, 例如用 [Netron](https://github.com/lutzroeder/Netron) 查看.
