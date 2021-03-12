# FAQ

----

- [FAQ](#faq)
  - [Single-Thread](#single-thread)
  - [Multiple-Thread](#multiple-thread)

----

## Single-Thread

1. How many kinds of inference modes are supported?
    - "float32" : default infer mode. If InferMode is not set, then the Forward engine use FLOAT mode to do inference.
    - "float16" : HALF infer mode
    - "int8" : INT8 infer mode. TensorRT layers support INT8 mode in default, but customized layers require more workload to support INT8 mode.
    - "int8_calib" : INT8 Calibration mode. If models include customized layers that support INT8 mode (like BERT), users should use this mode to generate at first a calibration cache file as CodeBook. After a valid calibration cache file is built, users can use INT8 mode to build a Forward engine for INT8 inference.

## Multiple-Thread

1. When using a Forward engine to do inference in multi-thread scenarios, why they run into Core Dump?
    - If you are using shared GPU services, you should check if different kinds of GPU devices are used at the same time, for example, T4, P4, and V100 are used in this shared GPU service. Inferences of Forward are based on TensorRT engines that they are built at specific GPU archs, so engines cannot work at multiple GPU archs.
    - Because [TensorRT engines are not thread-safe](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html#thread-safety), side effects of multi-threads like race condition should be handled. For example, mutex locks can be added around Load and Forward functions. There are some logs related to this problem as below:
        - `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 77 (an illegal memory access was encountered)`
        - `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 74 (misaligned address)`
    - The amount of GPU memory usages in multi-thread scenarios should not be larger than the maximum memory of the GPU device.