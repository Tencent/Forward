# Demo for building BERT model

> Use fwd_tf for BERT-tiny as example
0. 从 [BERT](https://github.com/google-research/bert/blob/master/README.md) 下载 BERT 模型. 此处以 BERT-tiny 为例.

1. 利用 [export_bert.py](../../python/bert_helpers/export_bert.py) 导出 `frozen_bert.pb`.

2. 参考 [CMake-Build](../../doc/cn/usages/cmake_build_CN.md) 构建 Forward 库.

3. 将 Forward-Python 库拷贝到当前目录.

5. 执行 `test_tf_bert.py` 脚本.

**Notice**: 可使用模型查看器来查看模型输入名和输出名，例如 [Netron](https://github.com/lutzroeder/Netron).
