# Demo for using Forward-Python

> Use fwd_torch as example

1. Build Forward-Python libraries refer to [CMake-Build](../../doc/en/usages/cmake_build.md).

2. Copy `forward.cpython.xxx*.so`(Linux) or `forward.xxx*.pyd`(Windows) in the `build/bin` into the workspace directory of Python project. For example, the directory is organized as:

```bash
---- workspace
   |
   -- test_forward.py
   |
   -- forward.cpython.xxx*.so
```

3. Use the build-used Python (Used in CMake-Build) to `import forward`.

**Notice**: The name of INPUT in models can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).
