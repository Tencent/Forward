# 常见问题

----

## 推理模式有哪些？

1. "float32" : 默认模式, 不设置 `mode` 参数时, 默认使用 float 浮点数计算推理模式. 
2. "float16" : 半精度计算推理模式. 
3. "int8" : INT8 量化计算推理模式. 自定义插件需自行支持, TensorRT原生层级默认支持 INT8 量化. 
4. "int8_calib" : INT8 量化码本生成模式. 当有自定义插件（如BERT）支持 INT8 推理时, 需预先用此模式生成一个对应的量化码本, 用于辅助之后 "int8" 模式的推理引擎构建.

## INT8 推理模式下无法构建

1. 检查 TensorRT 版本和 `ENABLE_DYNAMIC_BATCH` 开关。当 TensorRT version < 7.1.xx.xx and `ENABLE_DYNAMIC_BATCH=ON`时，INT8 推理模式的引擎无法构建。
2. 检查模型是否包含自定义插件层级。若存在自定义插件层级，则需要先用 `int8_calib` 模式预先构建 FLOAT 推理模式的引擎和 Calibration cache 文件。之后基于该 calibration cache 文件，构建 `int8` 推理模式的引擎。 **注意**: 自定义插件层级的实现需要针对 `int8_calib` 模式将输入输出数据类型设置为 Float。（可参考 BERT 插件的实现）

## 多线程使用场景下的 Core dumped

1. 当使用共享GPU服务时, 检查是否存在多种类 GPU 卡（如 T4, P4, V100）混用的情况. 因为 TensorRT Engine 与 GPU 架构强绑定, 无法进行多种类 GPU 卡混用的推理. 
2. 当在多线程并发调用 Forward 的场景时, Forward 的模型加载（Load）和推理（Forward）函数都不是 [线程安全](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html#thread-safety) 的, 考虑线程冲突的问题. 
    - 常见错误日志: `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 77 (an illegal memory access was encountered)`
    - 常见错误日志: `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 74 (misaligned address)`
3. 当在不同的机器集群内多线程使用 Forward 时（一个线程对应一个 Forward 对象）, 须注意多个 Engine 加载的模型总大小不能超过 GPU 显存上限. 

## cublasLt 相关的错误

1. 像 `[TRT] Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS \source\rtSafe\cublas\cublasLtWrapper.cpp:279`这类错误信息，一般与一个已知的 CUDA 10.2 中的 cubBLAS LT bug 有关. 它可以通过升级 CUDA 10.2 的补丁包或者用 TensorRT 的 API `config->setTacticSources()` 来禁用 cuBLAS Lt. (参考 [TensorRT Issue 1151](https://github.com/NVIDIA/TensorRT/issues/1151))