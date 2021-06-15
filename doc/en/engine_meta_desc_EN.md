# Engine Meta Description

> up-to-date 2021.06.15

Engine Meta is saved as a JSON file。

```json
{
    "infer_mode": 0,
    "max_batch_size": 1,
    "opt_batch_size": 1,
    "output_indices": [
        1,
        2,
        3,
        4
    ],
    "torch_module_path": "",
    "unused_input_indices": []
}

```

Engine Meta is a configuration file for loading Forward Engine.

- `infer_mode` : InferMode 
    - 0 = FLOAT
    - 1 = HALF
    - 2 = INT8
    - 3 = INT8_CALIB
    - Example: If Engine using Half mode, then it should be set  `1`.
- `opt_batch_size` : Optimal Batch Size.
- `max_batch_size` : Maximum Batch Size.
- `unused_input_indices` : unused input indices of tuple inputs of torch::Tensor.
    - Example: If tuple inputs of torch::Tensor is `(input_0, input_1, input_2)` but `input_2` is not used in Engine, then this attribute is set as `[2]`.
- `output_indices` : Output Binding Indices.
    - Example: If Engine has 4 outputs and 1 input, then it should be `[1, 2, 3, 4]`;
- `torch_module_path` : The path to TorchScript Module 模型路径. When some Torch operators cannot be converted to TensorRT operators, TorchModulePlugin is used to support these Torch operators. In this case, the original TorchScript Module is required such that Engine can load the module to rebuild a SubModule to forward.
