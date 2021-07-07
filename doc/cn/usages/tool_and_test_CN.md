# 工具与测试

## 辅助工具

为了分析网络构建的情况，在代码中有一些辅助工具，在构建/推理的过程中会运行并输出结果。相关的代码包括：
- `TrtForward::DumpNetwork`: 打印生成的 `TensorRT` 网络，打印信息包括层名、输入输出的维度信息和类型信息。需要注意的是，这里的类型信息不一定准确，这是 `TensorRT` 的已知 Bug，这一方法可以在 `TrtForward::Build` 中被调用，用户可以自行选择启用或关闭。
- `TrtCommon::SimpleProfiler`：打印各网络层使用的推理时间。使用此功能需开启宏 `TRT_INFER_ENABLE_PROFILING`，即在生成构建关系时需启用 `ENABLE_PROFILING` 。当 `TrtForward` 完成推理后，Forward 会给出相应的时间。

## 单元测试

对于大部分单独结点的转换，我们提供了单元测试组件以验证其转换的正确性。这些组件位于[单元测试目录](../../../source/unit_test)，其中：

| 文件名 | 内容 |
| :-----| :---- |
| `test_<platform>_nodes.h` | 测试对应平台下结点正确性 |

在 `unit_test_<platform>_helper.h` 中，我们提供了 `Test<Platform>Inference` 方法，用户可以根据此方法继续添加新的单元测试。

## 集成测试

对于常见的 CV / Bert / Recommender 模型，我们在[单元测试目录](../../../source/unit_test)中同样提供了相应的组件，用于测试模型转换的正确性。

| 文件名 | 内容 |
| :-----| :---- |
| `test_<platform>_vision.h` | 测试对应平台下 CV 模型的正确性 |
| `test_<platform>_bert.h` | 测试对应平台下 Bert 模型的正确性 |
| `test_torch_dlrm.h.h` | 测试 PyTorch 平台下 DLRM 推荐模型的正确性 |
| `test_tf_recommender.h` | 测试 TensorFlow 平台下部分推荐模型的正确性 |

## 性能测试

对于 CV 模型的性能指标，我们通过性能测试验证了其效果。在 `unit_test_<platform>_helper.h` 中，我们提供了 `Test<Platform>Time` 方法，用户可以根据此方法继续添加新的性能测试。