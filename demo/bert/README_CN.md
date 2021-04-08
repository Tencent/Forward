# Demo for building BERT model

**Notice**: 可使用模型查看器来查看模型输入名和输出名，例如 [Netron](https://github.com/lutzroeder/Netron).

## TensorFlow-BERT
0. 从 [BERT](https://github.com/google-research/bert/blob/master/README.md) 下载 BERT 模型. 此处以 BERT-tiny 为例.

1. 利用 [export_bert.py](../../python/bert_helpers/export_bert.py) 导出 `frozen_bert.pb`.

2. 参考 [CMake-Build](../../doc/cn/usages/cmake_build_CN.md) 构建 Forward 库. `-DENABLE_TENSORFLOW=ON`, `-DBUILD_PYTHON_LIB=ON`, `-DPYTHON_EXECUTABLE=$(which python3)`须要被声明.

3. 将 Forward-Python 库拷贝到当前目录.

4. 执行 `test_tf_bert.py` 脚本.

## Torch-BERT

0. 安装 `transformers` 和 `torch`.

1. 参考 [CMake-Build](../../doc/cn/usages/cmake_build_CN.md) 构建 Forward 库. `-DENABLE_TORCH=ON`, `-DBUILD_PYTHON_LIB=ON`, `-DPYTHON_EXECUTABLE=$(which python3)`须要被声明.

2. 将 Forward-Python 库拷贝到当前目录.

3. 执行 `test_torch_bert.py` 脚本.
