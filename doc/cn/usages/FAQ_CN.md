# 常见问题

----

## 常规使用

1. 推理模式有哪些？
    - 推理模式有:
        - "float32" : 默认模式, 不设置 `mode` 参数时, 默认使用 float 浮点数计算推理模式. 
        - "float16" : 半精度计算推理模式. 
        - "int8" : INT8 量化计算推理模式. 自定义插件需自行支持, TensorRT原生层级默认支持 INT8 量化. 
        - "int8_calib" : INT8 量化码本生成模式. 当有自定义插件（如BERT）支持 INT8 推理时, 需预先用此模式生成一个对应的量化码本, 用于辅助之后 "int8" 模式的推理引擎构建. 

## 多线程使用场景

1. 多线程场景下使用 Forward 的 Engine 进行推理, 会出现 Core Dump 的情况？
    - 当使用共享GPU服务时, 检查是否存在多种类 GPU 卡（如 T4, P4, V100）混用的情况. 因为 TensorRT Engine 与 GPU 架构强绑定, 无法进行多种类 GPU 卡混用的推理. 
    - 当在多线程并发调用 Forward 的场景时, Forward 的模型加载（Load）和推理（Forward）函数都不是 [线程安全](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html#thread-safety) 的, 考虑线程冲突的问题. 
        - 常见错误日志: `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 77 (an illegal memory access was encountered)`
        - 常见错误日志: `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 74 (misaligned address)`
    - 当在不同的机器集群内多线程使用 Forward 时（一个线程对应一个 Forward 对象）, 须注意多个 Engine 加载的模型总大小不能超过 GPU 显存上限. 