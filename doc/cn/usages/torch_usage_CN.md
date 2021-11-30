# Forward-PyTorch

----

- [Forward-PyTorch](#forward-pytorch)
  - [Prerequisites](#prerequisites)
  - [Pytorch Installation](#pytorch-installation)
  - [Pytorch JIT 模型导出](#pytorch-jit-模型导出)
  - [Build](#build)
  - [Dynamic batch inputs](#dynamic-batch-inputs)
    - [CMake options](#cmake-options)
    - [Setting](#setting)
  - [Cpp Example](#cpp-example)
    - [Cpp INT8 Example](#cpp-int8-example)
    - [Cpp BERT INT8 Example](#cpp-bert-int8-example)
  - [Python Example](#python-example)
    - [Python INT8 Example](#python-int8-example)
    - [Python BERT-INT8 Example](#python-bert-int8-example)
  - [使用手工设定scale进行量化](#使用手工设定scale进行量化)
    - [Cpp](#cpp)
    - [Python](#python)

----

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- PyTorch >= 1.7.0

 __Note: 使用 Conda 或者 pip 安装的 pytorch 预编译版本, 所使用的 CUDA 必须和系统环境下的 CUDA toolkit 版本一致, 否则在使用 python lib 退出时可能会出现 Segmentation fault.__

## Pytorch Installation

1. Install from pip

``` sh
pip3 install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

2. Install from whl
Download whl from [here](https://download.pytorch.org/whl/torch_stable.html), for example python 3.6 + CPU:

``` sh
https://download.pytorch.org/whl/cpu/torch-1.7.1%2Bcpu-cp36-cp36m-linux_x86_64.whl
```

## Pytorch JIT 模型导出

Forward 加载的模型须是 torch 的 JIT 模型, 且必须是 cpu 的模型导出. 可参考以下案例
```python
import torch

def TracedModelFactory(file_name, traced_model):
    traced_model.save(file_name)
    traced_model = torch.jit.load(file_name)
    print("filename : ", file_name)
    print(traced_model.graph)

class ArithmeticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b):
        a1 = torch.add(a, b)
        b1 = torch.rsub(a, b)
        c1 = torch.sub(a, b)
        d1 = torch.mul(a, b)
        return a1, b1, c1, d1
    
a = torch.randn(4)
b = torch.randn(4)
model = ArithmeticModule()
model.eval() # 模型必须是 Eval 推理模式
model.cpu()  # 模型必须是 Cpu 参数
traced_model = torch.jit.trace(model, (a, b))

TracedModelFactory("arithmetic.pth", traced_model)
```

## Build

```bash
mkdir build
cd build
cmake ..                               \ 
-DTensorRT_ROOT="path/to/TensorRT"     \ 
-DCMAKE_PREFIX_PATH="path/to/LibTorch" \ 
-DENABLE_TENSORFLOW=OFF                \ 
-DENABLE_TORCH=OFF                     \ 
-DENABLE_KERAS=ON                      \ 
-DENABLE_ONNX=OFF                      \ 
-DENABLE_UNIT_TESTS=ON                 \ 
-DBUILD_PYTHON_LIB=OFF                 \ 
-DPYTHON_EXECUTABLE="/path/to/python3"

make -j
```

## Dynamic batch inputs

**注意**: 仅当 TensorRT 版本大于 7.1.xx.xx 时，才可在 INT8 推理模式下使用动态批量输入功能。

### CMake options

- 当动态批量功能开关打开 `-DENABLE_DYNAMIC_BATCH=ON` 时, Engine 可用 1 < `batch_size` < `max_batch_size` 的任意 batch_size 来进行推理. 另外, 也可以设置一个 `opt_batch_size`, 则 Engine 将会根据此批量大小进行针对性优化.

### Setting

- `max_batch_size`: 构建 Engine 时, `dummy_input` 中的 `batch_size` 将会被设置为 `max_batch_size`.
- `opt_batch_size`: 构建 Engine 时, 需要显式调用接口来设置 `opt_batch_size`. 如果没有被设置, 则 `opt_batch_size` 将与 `max_batch_size` 保持一致. 
  - c++ 调用接口: `tf_builder.SetOptBatchSize(opt_batch_size)`
  - python 调用接口: `tf_builder.set_opt_batch_size(opt_batch_size)`

## Cpp Example

```c++
// 1. 构建 Engine
fwd::TorchBuilder torch_builder;

std::string model_path = "path/to/jit/module";
const std::string infer_mode = "float32"; // float32 / float16 / int8_calib / int8
const c10::DeviceType device = c10::kCPU; // c10::kCPU / c10::kCUDA
// 伪输入的数据类型和维度须与真实输入保持一致 
std::vector<c10::IValue> dummy_inputs {torch::randn({1, 3, 224, 224}, device)};

torch_builder.SetInferMode(infer_mode);
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);

// // 带名字的伪输入构建方式
//
// std::unordered_map<std::string, c10::IValue> dummy_inputs;
// // 输入名可用 Netron 查看模型获知
// dummy_inputs["input"] =  torch::randn({1, 3, 224, 224}, device); 
//

std::vector<c10::IValue> inputs = dummy_inputs; 
bool need_save = true;
if (!need_save){
    std::vector<torch::Tensor> outputs = torch_engine->Forward(inputs);
    // std::vector<torch::Tensor> outputs = torch_engine->ForwardWithName(dummy_inputs); 

}else{
    std::string engine_file = "path/to/out/engine";
    torch_engine->Save(engine_file);

    fwd::TorchEngine torch_engine;
    torch_engine.Load(engine_file);
    std::vector<torch::Tensor> outputs = torch_engine.Forward(inputs);
    // std::vector<torch::Tensor> outputs = torch_engine.ForwardWithName(dummy_inputs);
}
```

### Cpp INT8 Example

```c++
#include "common/trt_batch_stream.h"
// 继承实现数据批量获取工具类
class ImgBatchStream : public IBatchStream {
    // 是否还有batch
    bool next() override {...};
    // 获取输入, 一个输入为一个vector<float>
    std::vector<std::vector<float>> getBatch() override {...};
    // 返回一个batch的大小
    int getBatchSize() const override {...};
    // 返回getBatch中每个输入的长度
    std::vector<int64_t> size() const override {...};
}
std::shared_ptr<IBatchStream> ibs = std::make_shared<ImgBatchStream>();
// 创建量化器实例, 参数分别为BatchStream, 缓存名, 量化算法名[entropy | entropy_2 | minmax]
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "entropy");

fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8"); 
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);
```

### Cpp BERT INT8 Example

- 相对于普通 INT8 的构建, BERT INT8 需要提前用 `int8_calib` 模式生成一个 calibrator 码本, 然后再用 `int8` 模式自动加载生成的码本来构建 int8 推理引擎. 

```c++
#include "common/trt_batch_stream.h"
// 继承实现数据批量获取工具类, 可参考 test_batch_stream.h 里面的 BertStream
class TestBertStream : public fwd::IBatchStream {
 public:
  TestBertStream(int batch_size, int seq_len, int emb_count)
      : mBatchSize(batch_size),
        mSeqLen(seq_len),
        mSize(batch_size * seq_len),
        mEmbCount(emb_count) {
    mData.resize(3, std::vector<int>(mSize));
  }

  bool next() override { return mBatch < mBatchTotal; }

  std::vector<const void*> getBatch() override {
    if (mBatch < mBatchTotal) {
      ++mBatch;
      std::vector<const void*> batch;

      for (int i = 0; i < mSize; i++) mData[0].push_back(rand() % mEmbCount);
      batch.push_back(mData[0].data());

      for (int i = 0; i < mBatchSize; i++) {
        int rand1 = rand() % (mSeqLen - 1) + 1;
        for (int j = 0; j < rand1; j++) mData[1][i * mSeqLen + j] = 1;
        for (int j = rand1; j < mSeqLen; j++) mData[1][i * mSeqLen + j] = 0;
      }
      batch.push_back(mData[1].data());

      for (int i = 0; i < mSize; i++) {
        mData[2][i] = rand() % 2;
      }
      batch.push_back(mData[2].data());

      return batch;
    }
    return {{}};
  }
  int getBatchSize() const override { return mBatchSize; };
  std::vector<int64_t> bytesPerBatch() const override {
    std::vector<int64_t> bytes(3, mSeqLen * sizeof(int));
    return bytes;
  }

 private:
  int64_t mSize;
  int mBatch{0};
  int mBatchTotal{500};
  int mEmbCount{0};
  int mBatchSize{0};
  int mSeqLen{0};
  std::vector<std::vector<int>> mData;  // input_ids, input_mask, segment_ids
};

std::shared_ptr<IBatchStream> ibs = std::make_shared<BertBatchStream>();
// 创建量化器实例, 参数分别为BatchStream, 缓存名, 量化算法名用 minmax
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "minmax");

// 构建码本
fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8_calib");
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);

// 使用码本构建推理引擎
fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8");
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);
```

## Python Example

构建成功后,  须要将 `build/bin` 目录下的 `forward*.so`(Linux) 或 `forward*.pyd`(Windows) 拷贝到 Python 工作目录下.

``` python
import torch
import forward

# 1. 构建 Engine
builder = forward.TorchBuilder()
dummy_inputs = torch.randn(1, 3, 224, 224) 
infer_mode = 'float32'  #  float32 / float16 / int8_calib / int8

builder.set_mode(infer_mode)
engine = builder.build('path/to/jit/module', dummy_inputs)

# （可选）使用带名字的伪输入接口构建, 输入名可用 Netron 模型查看器查看
'''
dummy_inputs = {"input" : torch.randn(1, 3, 224, 224) } 
engine = builder.build_with_name('path/to/jit/module', dummy_inputs)
'''

need_save = True
if need_save:
    engine_path = 'path/to/engine'
    engine.save(engine_path)

    engine = forward.TorchEngine()
    engine.load(engine_path)

# 2. 执行推理
inputs = torch.randn(1, 3, 224, 224).cuda() # network input tensors
outputs = engine.forward(inputs)  # 如果输入是 cuda tensor, 输出也是 cuda tensor. 如果输入是 cpu tensor,  输出也是 cpu tensor

# v2.5 之后的新功能
'''
outputs = engine.forward_with_name(dummy_inputs) # dummy_inputs 须为 dict 类型, 输入名通过 Netron 查看模型获知
'''
```

### Python INT8 Example

```python
import forward
import numpy as np
# 1. 继承实现数据提供工具类
class MBatchStream(forward.IPyBatchStream):
    def __init__(self):
        forward.IPyBatchStream.__init__(self) # 必须调用父类的初始化方法
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

    def getNumpyBatch(self):
        return [np.random.randn(1*24*24*3)]

bs = MBatchStream()
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.MINMAX_CALIBRATION)

builder = forward.TorchBuilder()
builder.set_calibrator(calibrator)

# 构建 engine
builder.set_mode("int8") 
engine = builder.build('path/to/jit/module', dummy_inputs)
```

### Python BERT-INT8 Example

- 相对于普通 INT8 的构建, BERT INT8 需要提前用 `int8_calib` 模式生成一个 calibrator 码本, 然后再用 `int8` 模式自动加载生成的码本来构建 int8 推理引擎. 

```python
import forward
import bert_helpers.tokenization
import bert_helpers.data_preprocessing as dp
import numpy as np
# 1. 继承实现数据提供工具类
class BertBatchStream(forward.IPyBatchStream):
    def __init__(self, squad_json, vocab_file, cache_file, batch_size, max_seq_length, num_inputs):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        forward.IPyBatchStream.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = dp.read_squad_json(squad_json)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        self.num_inputs = num_inputs
        self.tokenizer = tokenization.BertTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.doc_stride = 128
        self.max_query_length = 64
        self.maxbatch = 500

        # Allocate enough memory for a whole batch.
        #self.device_inputs = [cuda.mem_alloc(self.max_seq_length * trt.int32.itemsize * self.batch_size) for binding in range(3)]

    def free(self):
        # for dinput in self.device_inputs:
        #    dinput.free()
        return

    def next(self):
        if self.current_index < self.num_inputs:
            return True
        return False

    def bytesPerBatch(self):
        s = self.batch_size * self.max_seq_length * 4
        return [s, s, s]

    def getBatchSize(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    # def get_batch(self, names):
    def getNumpyBatch(self):
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(
                self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(
                current_batch, self.batch_size))

        input_ids = []
        segment_ids = []
        input_mask = []
        for i in range(self.batch_size):
            example = self.data[self.current_index + i]
            features = dp.convert_example_to_features(
                example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            if len(input_ids) and len(segment_ids) and len(input_mask):
                input_ids = np.concatenate((input_ids, features[0].input_ids))
                segment_ids = np.concatenate(
                    (segment_ids, features[0].segment_ids))
                input_mask = np.concatenate(
                    (input_mask, features[0].input_mask))
            else:
                input_ids = np.array(features[0].input_ids, dtype=np.int32)
                segment_ids = np.array(features[0].segment_ids, dtype=np.int32)
                input_mask = np.array(features[0].input_mask, dtype=np.int32)
        self.current_index += self.batch_size
        self.current_data = [input_ids, input_mask, segment_ids]
        return self.current_data

bs = BertBatchStream()
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.MINMAX_CALIBRATION)

# 2. 构建码本
builder = forward.TorchBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8_calib") //（可选）若不设置, 则默认为 FP32
engine = builder.build('path/to/jit/module', dummy_inputs)
# build_with_name 接口
'''
dummy_inputs = {"input_ids" : xxx , "attention_mask" : xxx, "input" : xxx } // 模型输入名可通过 Netron 查看模型获知
engine = builder.build_with_name('path/to/jit/module', dummy_inputs)
'''

# 3. 使用码本构建引擎
builder = forward.TorchBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8") //（可选）若不设置, 则默认为 FP32
engine = builder.build('path/to/jit/module', dummy_inputs)
# engine = builder.build_with_name('path/to/jit/module', dummy_inputs) 
```

## 使用手工设定scale进行量化

使用scale量化不需要实现IBatchStream工具类, 但需要提供每层的scale

scale文件可由非手动量化的缓存文件修改得：

格式为每行为需要设定的层输出名: 缩放scale值`TensorName: float_scale`

```
TRT-7000-EntropyCalibration
INPUT0:4.6586
(Unnamed Layer* 2) [Activation]_output:10.8675
(Unnamed Layer* 3) [Pooling]_output:11.1173
...
```

### Cpp

```c++
// 1. 准备量化器
const std::string cacheTableName = "calibrator.cache"; // calib缓存文件名
const std::string algo = "entropy"; // 量化算法名[entropy | entropy_2 | minmax]
int batch_size = 1;    // batch size
// 创建量化器实例
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(cacheTableName , algo , batch_size);
const std::string customized_cache_file = "path/to/scale_file.txt";
calib->setScaleFile(customized_cache_file);

// 2. build engine
fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8");
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);
```

### Python

```python
# 1. 准备量化器
cacheTableName = "calibrator.cache"
algo = "entropy" # algo = forward.ENTROPY_CALIBRATION | "entropy_2" | "minmax"
batch_size = 1
calib = forward.TrtInt8Calibrator(cacheTableName , algo , batch_size)

customized_cache_file = "path/to/scale_file.txt"
calib.set_scale_file(customized_cache_file)

# 2. 构建 engine
builder.set_calibrator(calibrator)
builder.set_mode("int8") 
engine = builder.build('path/to/jit/module', dummy_inputs)
```
