# FAQ

----

- [FAQ](#faq)
  - [Infer modes](#infer-modes)
  - [engine with INT8 mode cannot be built](#engine-with-int8-mode-cannot-be-built)
  - [Core dumped in multi-thread scenarios](#core-dumped-in-multi-thread-scenarios)
  - [cublasLt related errors](#cublaslt-related-errors)

----

## Infer modes

1. "float32" : default infer mode. If InferMode is not set, then the Forward engine use FLOAT mode to do inference.
2. "float16" : HALF infer mode
3. "int8" : INT8 infer mode. TensorRT layers support INT8 mode in default, but customized layers require more workload to support INT8 mode.
4. "int8_calib" : INT8 Calibration mode. If models include customized layers that support INT8 mode (like BERT), users should use this mode to generate at first a calibration cache file as CodeBook. After a valid calibration cache file is built, users can use INT8 mode to build a Forward engine for INT8 inference.

## engine with INT8 mode cannot be built

1. Check TensorRT version and `ENABLE_DYNAMIC_BATCH`, INT8 mode cannot work when TensorRT version < 7.1.xx.xx and `ENABLE_DYNAMIC_BATCH=ON`.
2. Check if the model has customized plugin layer. If customized plugin exists, the engine with `int8_calib` mode should be built first for a valid calibration cache file. With this cache file, the engine with `int8` can successfully be built. **Notice**: the implementation of customized plugin should explicitly set the data type of inputs and outputs as Float for `int8_calib` mode. (Refer to BERT plugins)

## Core dumped in multi-thread scenarios

1. If you are using shared GPU services, you should check if different kinds of GPU devices are used at the same time, for example, T4, P4, and V100 are used in this shared GPU service. Inferences of Forward are based on TensorRT engines that they are built at specific GPU archs, so engines cannot work at multiple GPU archs.
2. Because [TensorRT engines are not thread-safe](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html#thread-safety), side effects of multi-threads like race condition should be handled. For example, mutex locks can be added around Load and Forward functions. There are some logs related to this problem as below:
    - `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 77 (an illegal memory access was encountered)`
    - `[TRT] engine.cpp (902) - Cuda Error in executeInternal: 74 (misaligned address)`
3. The amount of GPU memory usages in multi-thread scenarios should not be larger than the maximum memory of the GPU device.

## cublasLt related errors

1. Error message like `[TRT] Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS \source\rtSafe\cublas\cublasLtWrapper.cpp:279` is almost related to a known cubBLAS LT bug in CUDA 10.2. You can fix it either by upgrading to a newer patch version of 10.2, or using the work-around mentioned here. Since you're using the API you can use  `config->setTacticSources()`  to disable cuBLAS LT. (Refer to [TensorRT Issue 1151](https://github.com/NVIDIA/TensorRT/issues/1151))