# 推理引擎使用方法

## 方法一：通过各模型的 `Engine` 类接口进行推理

使用模型对应的 `TfBuilder` / `TorchBuilder` / `KerasBuilder` 类构建引擎后，会立刻返回该模型的 `TfEngine` / `TorchEngine` / `KerasEngine`，这些 `Engine` 可以被保存到文件中，或者直接使用对应模型平台结构的数据进行推理（如 `TF_Tensor` 或 `torch::Tensor`），在实现上会将这些输入数据转换成 `TensorRT` 的标准输入类型，然后传入 `TrtForward` 中进行推理。这种方式适合通过 Python 调用 Forward 进行推理。更多 Python 示例[参考此处](../../../demo/fwd_py)。

```python
import forward
import numpy as np

# 1. 构建 Engine
builder = forward.TfBuilder()
batch_size = 16
infer_mode = 'float32'

dummy_input = {"input_ids" : np.ones([batch_size , 48], dtype='int32'), 
               "input_mask" : np.ones([batch_size , 48], dtype='int32'),
               "segment_ids" : np.ones([batch_size , 48], dtype='int32')}

builder.set_mode(infer_mode); 
tf_engine = builder.build('bert_model.pb', dummy_input)

# 2. 调用 Forward 推理
need_save = True
if need_save:
    # save engine
    engine_path = 'path/to/out/engine'
    tf_engine.save(engine_path)

    # load saved engine
    tf_engine = forward.TfEngine()
    tf_engine.load(engine_path)

inputs = dummy_input
outputs = tf_engine.forward(inputs) 
```

## 方法二：通过统一的 `TrtForwardEngine` 类接口进行推理

`TrtForwardEngine` 类中的 `Load` 函数能够直接读取由上述方法保存得到的推理引擎，并通过调用 `TrtForwardEngine::ForwardWithName` 函数直接进行推理，它的输入要求是自定义的 `Tensor` 结构类型。采用这种方式的好处在于它剥离了原始模型的相关结构，使得库更轻便，同时也能够兼容各种模型转换得到的推理引擎。因此，这种方式适用于服务端进行推理。更多 C++ 示例[参考此处](../../../demo/fwd_cpp)。

```cpp
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>

#include "cuda_helper.h"
#include "trt_fwd_engine.h"

int main() {
  fwd::TrtForwardEngine fwd_engine;

  // Step 1: Update the path to pb model
  std::string engine_path = "../data/softmax.pb.engine";

  // Step 2: Load engine
  if (!fwd_engine.Load(engine_path)) {
    std::cerr << "Engine Load failed on " << engine_path << std::endl;
    return -1;
  }

  // Step 3: Prepare inputs
  fwd::IOMappingVector feed_dict;
  fwd::IOMappingVector outputs;

  fwd::NamedTensor input;
  input.name = "input_11";
  auto shape = std::vector<int>{16, 12, 24, 3};
  input.tensor.dims = shape;
  auto volume = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
  PinnedVector<float> data;
  data.Resize(volume);
  memset(data.Data(), 0, sizeof(float) * volume);
  input.tensor.data = data.Data();
  input.tensor.data_type = fwd::DataType::FLOAT;
  input.tensor.device_type = fwd::DeviceType::CPU;
  feed_dict.push_back(input);

  // Step 4: Forward with engine
  if (!fwd_engine.ForwardWithName(feed_dict, outputs)) {
    std::cerr << "Engine forward error! " << std::endl;
    return -1;
  }

  // Step 5: Organize outputs
  PinnedVector<float> h_outputs;
  const auto& out_shape = outputs[0].tensor.dims;
  auto out_volume =
      std::accumulate(out_shape.cbegin(), out_shape.cend(), 1, std::multiplies<int>());
  h_outputs.Resize(out_volume);
  MemcpyDeviceToHost(h_outputs.Data(), reinterpret_cast<float*>(outputs[0].tensor.data),
                     out_volume);

  auto* data_ptr = h_outputs.Data();
  std::cout << "Print Head 10 elements" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << *(data_ptr + i) << ", ";
  }

  std::cout << std::endl << "Test Engine finished." << std::endl;
  return 0;
}
```