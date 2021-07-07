# Inference Engine Usage

## Method 1: inference through the `Engine` class interface of each model

After building the engine using the model class (`TfBuilder` / `TorchBuilder` / `KerasBuilder`), Forward will return the corresponding engine (`TfEngine` / `TorchEngine` / `KerasEngine`). These engines can be saved into files for later use or can be used directly to inference with proper input data types (`TF_Tensor` 或 `torch::Tensor`). Under the implementation, Forward will convert these types into the standard input data format supported by `TensorRT` and then pass them input into `TrtForward` to inference. This method is suitable for inference by calling Forward API through Python. More Python examples refer to [demo/fwd_py](../../../demo/fwd_py).

```python
import forward
import numpy as np

# 1. Build engine
builder = forward.TfBuilder()
batch_size = 16
infer_mode = 'float32'

dummy_input = {"input_ids" : np.ones([batch_size , 48], dtype='int32'), 
               "input_mask" : np.ones([batch_size , 48], dtype='int32'),
               "segment_ids" : np.ones([batch_size , 48], dtype='int32')}

builder.set_mode(infer_mode); 
tf_engine = builder.build('bert_model.pb', dummy_input)

# 2. Call Forward inference
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

## Method 2: inference through the unified `TrtForwardEngine` class interface

The `Load` function implemented in the `TrtForwardEngine` class can directly load the inference engine saved in Method 1. After loading, `TrtForwardEngine::ForwardWithName` can be called directly for inference. It requires a customized `Tensor` data structure. The advantage of adopting this method is that it strips away the relevant structure of the original model, making the library more portable, and it is also compatible with the inference engine obtained from various model conversions. Therefore, this method is suitable for inference on the server-side. More C++ examples refer to [demo/fwd_cpp](../../../demo/fwd_cpp).

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