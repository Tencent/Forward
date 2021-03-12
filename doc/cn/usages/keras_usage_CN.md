# Forward-Keras

----

- [Forward-Keras](#forward-keras)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [C++ Example](#cpp-example)
    - [C++ INT8 Example](#cpp-int8-example)
  - [Python Example](#python-example)
    - [Python INT8 Example](#python-int8-example)

----

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 6.0.1.5,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.10.1
- GCC >= 5.4.0, ld >= 2.26.1
- (Keras) HDF 5

## Build

```bash
mkdir build
cd build
cmake .. \
-DTensorRT_ROOT=/path/to/TensorRT \  
-DENABLE_LOGGING=OFF \  
-DENABLE_PROFILING=OFF \ 
-DENABLE_DYNAMIC_BATCH=OFF \ 
-DBUILD_PYTHON_LIB=ON \ 
-DPYTHON_EXECUTABLE=/path/to/python3 \ 
-DENABLE_TORCH=OFF \
-DENABLE_TENSORFLOW=OFF \
-DENABLE_KERAS=ON
make -j
```

## Cpp Example

```c++
KerasBuilder keras_builder;

// 1. 构建 Engine
std::string model_path = "path/to/keras.h5";
const int input_batch_size = 1; 
const string infer_mode = "float32";

keras_builder.SetInferMode(infer_mode); 
std::shared_ptr<KerasEngine> engine = keras_builder.Build(model_path, input_batch_size);

bool need_save = true;
if (need_save) {
  std::string engine_file = "path/to/out/engine";
  engine->Save(engine_file);
  eninge = std::make_shared<KerasEngine>();
  engine->Load(engine_file);
}

// 2. 执行推理
// feed inputs data and input shapes
std::vector<std::vector<void*>> inputs;
std::vector<std::vector<int>> input_shapes;
std::vector<std::vector<void*>> outputs;
std::vector<std::vector<int>> output_shapes;
keras_engine->forward(inputs, input_shapes, outputs, output_shapes);

// 注意：默认输出位于 Device memory
// 可选： 将输出拷贝到 Host memory
cudaMemcpy(xxx, outputs[i], cudaMemcpyDeviceToHost);
```

### Cpp INT8 Example

```c++
#include "common/trt_batch_stream.h"
// 继承实现数据批量获取工具类
class ImgBatchStream : public IBatchStream {
    // 是否还有 Batch
    bool next() override {...};
    // 获取下一个输入
    std::vector<std::vector<float>> getBatch() override {...};
    // 获取 batch size
    int getBatchSize() const override {...};
    // 返回 getBatch 中每个输入的大小
    std::vector<int64_t> size() const override {...};
}
std::shared_ptr<IBatchStream> ibs = std::make_shared<ImgBatchStream>();
// 创建量化器实例, 参数分别为BatchStream, 缓存名, 量化算法名[entropy | entropy_2 | minmax]
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "entropy");
KerasBuilder keras_builder;
keras_builder.SetCalibrator(calib);

keras_builder.SetInferMode( 'int8');
std::shared_ptr<KerasEngine> engine = keras_builder.Build(model_path, input_batch_size);
```

## Python Example

``` python
import forward
import numpy as np

# 1. 构建 Engine
builder = forward.KerasBuilder()
infer_mode = 'float32' # Infer Mode: float32 / float16 / int8_calib / int8
batch_size = 1
max_workspace_size = 1<<32

builder.set_mode(infer_mode) 
engine = builder.build('path/to/kears.h5', batch_size)

need_save = True
if need_save:
    engine_path = 'path/to/out/engine'
    engine.save(engine_path)
    engine = forward.KerasEngine()
    engine.load(engine_path)

# 2. 执行推理
inputs = np.random.randn(1, 24, 24, 3) 
outputs = engine.forward([inputs]) # list_type output
```

### Python INT8 Example

```python
import forward
import numpy as np
# 1. 继承实现数据提供工具类
class MBatchStream(forward.IBatchStream):
    def __init__(self):
        forward.IBatchStream.__init__(self) # 必须调用父类的初始化方法
        self.batch = 0
        self.maxbatch = 500 

    def next(self):
        if self.batch < self.maxbatch:
            self.batch += 1
            return True
        return False

    def getBatchSize(self):
        return 1

    def size(self):
        return [1*24*24*3]

    def getBatch(self):
        return [np.random.randn(1*24*24*3)]

bs = MBatchStream()
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.ENTROPY_CALIBRATION)
builder = forward.KerasBuilder()
builder.setCalibrator(calibrator )

# 2. 构建 Engine
builder.set_mode('int8')
engine = builder.build('path/to/kears.h5', batch_size)
```
