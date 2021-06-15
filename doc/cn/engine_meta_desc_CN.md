# Engine Meta 描述

> up-to-date 2021.06.15

Engine Meta 保存为 JSON 格式。

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

每一项代表一种 Engine 生成时的配置参数，用以验证 Engine 的正确性。以下为每一项的参数说明：

- `infer_mode` : 推理模式 InferMode，填写以下数字
    - 0 = FLOAT
    - 1 = HALF
    - 2 = INT8
    - 3 = INT8_CALIB
    - 案例：若 Engine 为 Half，则设置为 `1`。
- `opt_batch_size` : 最优 Batch Size，填写 Optimal Batch Size 对应的数字。
- `max_batch_size` : 最大 Batch Size，填写 Max Batch Size 对应的数字。
- `unused_input_indices` : 未使用的输入序号。
    - 案例：若 Engine 有 2 个输入，原始模型也是 2 个输入，则填写为 0。
- `output_indices` : 输出的 Binding 序号顺序。若输出序号非乱序，则从输入的个数开始按顺序填写。
    - 案例：若 Engine 有 1 个输入，4个输出，输出皆按顺序输出，则该行设置为：`[1, 2, 3, 4]`。因为 `0` 为输入 binding，顺序输出 binding 为 `[1, 2, 3, 4]`。
- `torch_module_path` : TorchScript Module 模型路径。当某些 Torch Operators 转换成 TensorRT 算子失败时，将会用 TorchModulePlugin 来兼容该算子。届时，将需要指定原始的 TorchScriptModule 路径，用于在 TensorRT 推理时，TorchModulePlugin 加载该模型来重构 SubModule 用于推理。
