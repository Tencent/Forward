# Forward-PyTorch 

----

- [Forward-PyTorch](#forward-pytorch)
  - [Prerequisites](#prerequisites)
  - [Pytorch Installation](#pytorch-installation)
  - [Export PyTorch JIT models](#export-pytorch-jit-models)
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
  - [Customized Calibration cache file](#customized-calibration-cache-file)
    - [Cpp](#cpp)
    - [Python](#python)

----

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1
- (Pytorch) pytorch == 1.3.1 and pytorch == 1.7.1

 __Note: When using PyTorch that pre-installed by Conda or pip, its CUDA version should be as the same as the version of CUDA toolkit installed in the operating system. Otherwise, Forward-Python library may cause Segmentation fault.__

## Pytorch Installation

1. Install from pip

``` sh
pip3 install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

2. Install from whl
Download whl from [here](https://download.pytorch.org/whl/torch_stable.html), for example python 3.6 + CUDA 10.0:

``` sh
https://download.pytorch.org/whl/cu100/torch-1.3.1%2Bcu100-cp36-cp36m-linux_x86_64.whl
https://download.pytorch.org/whl/cu100/torchvision-0.4.2%2Bcu100-cp36-cp36m-linux_x86_64.whl
```

## Export PyTorch JIT models

Only PyTorch JIT models are supported by Forward, and they must be `cpu()` and `eval()`. PyTorch JIT models can be exported as following steps:

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
model.eval() # required
model.cpu()  # required
traced_model = torch.jit.trace(model, (a, b))

TracedModelFactory("arithmetic.pth", traced_model)
```

## Build

``` sh
mkdir build
cd build
cmake ..  \
-DTensorRT_ROOT=/path/to/TensorRT \  
-DENABLE_LOGGING=OFF \  
-DENABLE_PROFILING=OFF \  
-DENABLE_DYNAMIC_BATCH=OFF \  
-DBUILD_PYTHON_LIB=ON \ 
-DPYTHON_EXECUTABLE=/path/to/python3 \  
-DENABLE_TORCH=ON \
-DENABLE_TENSORFLOW=OFF \
-DENABLE_KERAS=OFF \

make -j
```

## Dynamic batch inputs

**Notice**: Dynamic batch inputs support INT8 mode ONLY when TensorRT version > 7.1.xx.xx.

### CMake options

- Set `-DENABLE_DYNAMIC_BATCH=ON` to support dynamic batch inputs, then `batch_size` between 1 and `max_batch_size` can be valid during inference periods. In this case, Forward engines will be optimized according to `opt_batch_size`.

### Setting

- `max_batch_size`: When building engines, the `batch_size` of `dummy_input` is set as `max_batch_size` for Forward engine.
- `opt_batch_size`: When building engines, `opt_batch_size` should be explicitly set by following interfaces. If it is not set, it will be set as the same as `max_batch_size` in `dummy_input`.
  - c++ interface: `torch_builder.SetOptBatchSize(opt_batch_size)`
  - python interface: `torch_builder.set_opt_batch_size(opt_batch_size)`

## Cpp Example

```c++
fwd::TorchBuilder torch_builder;

std::string model_path = "path/to/jit/module";
const std::string infer_mode = "float32"; // float32 / float16 / int8_calib / int8
const c10::DeviceType device = c10::kCPU; // c10::kCPU / c10::kCUDA
// DataTypes and Dimensions of dummy_input should be the same as those of real inputs 
std::vector<c10::IValue> dummy_inputs {torch::randn({1, 3, 224, 224}, device)};
// // build with input names
//
// std::unordered_map<std::string, c10::IValue> dummy_inputs;
// // input names can be viewed by model viewers like Netron
// dummy_inputs["input"] =  torch::randn({1, 3, 224, 224}, device); 
//

torch_builder.SetInferMode(infer_mode);
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);

std::vector<torch::jit::IValue> inputs = dummy_inputs;
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

// Inherit from IBatchStream and implement override functions
class TestBatchStream : public IBatchStream {
    // check if has next batch
    bool next() override {...};
    // get next batch inputs
    std::vector<std::vector<float>> getBatch() override {...};
    // get batch size of next batch inputs
    int getBatchSize() const override {...};
    // get volume size of next batch inputs
    std::vector<int64_t> size() const override {...};
}
std::shared_ptr<IBatchStream> ibs = std::make_shared<ImgBatchStream>();
// create TrtInt8Calibrator, algorithm can be [entropy | entropy_2 | minmax]
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "entropy");

fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8"); 
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);
```

### Cpp BERT INT8 Example

- Unlike building INT8 engines for normal models, building INT8 engines for BERT models has to use `int8_calib` mode to generate a calibration cache as CodeBook at first, and then use `int8` mode to build engines with the calibration cache file.

```c++
#include "common/trt_batch_stream.h"
// Inherit IBatchStream
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
// use 'minmax' algorithm
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "minmax");

// build with int8_calib mode to generate a calibration cache file
fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8_calib");
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);

// build int8 engine with the saved calibration cache file
fwd::TorchBuilder torch_builder;
torch_builder.SetCalibrator(calib);
torch_builder.SetInferMode("int8");
std::shared_ptr<fwd::TorchEngine> torch_engine = torch_builder.Build(model_path, dummy_inputs);
```

## Python Example

After building Forward project, `forward*.so`(Linux) or `forward*.pyd`(Windows) in `build/bin` directory should be copied into WORK_DIR of Python project.

``` python
import torch
import forward

# 1. build Engine
builder = forward.TorchBuilder()
dummy_inputs = torch.randn(1, 3, 224, 224) 
infer_mode = 'float32'  #  float32 / float16 / int8_calib / int8

builder.set_mode(infer_mode)
engine = builder.build('path/to/jit/module', dummy_inputs)

# build_with_name interface, input names can be viewed by model viewer like Netron
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

# 2. inference
inputs = torch.randn(1, 3, 224, 224).cuda() # network input tensors
# if inputs are CUDA tensors, then outputs are also CUDA tensors.
# if inputs are CPU tensors, then outputs are also CPU tensors.
outputs = engine.forward(inputs)

# forward_with_name interface
'''
outputs = engine.forward_with_name(dummy_inputs) # dict_type inputs
'''
```

### Python INT8 Example

```python
import forward
import numpy as np
# 1. Inherit forward.IPyBatchStream
class MBatchStream(forward.IPyBatchStream):
    def __init__(self):
        forward.IPyBatchStream.__init__(self) # required
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

# build engine
builder.set_mode("int8") 
engine = builder.build('path/to/jit/module', dummy_inputs)
```

### Python BERT-INT8 Example

- Unlike building INT8 engines for normal models, building INT8 engines for BERT models has to use `int8_calib` mode to generate a calibration cache as CodeBook at first, and then use `int8` mode to build engines with the calibration cache file.

```python
import forward
import bert_helpers.tokenization
import bert_helpers.data_preprocessing as dp
import numpy as np
# 1. Inherit from forward.IPyBatchStream
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

bs = BertBatchStream(...)
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.MINMAX_CALIBRATION)

# 2. generate calibration cache as CodeBook
builder = forward.TorchBuilder()
builder.set_calibrator(calibrator)builder.set_mode("int8_calib") //（可选）若不设置, 则默认为 FP32
engine = builder.build('path/to/jit/module', dummy_inputs)
# build_with_name interface
'''
dummy_inputs = {"input_ids" : xxx , "attention_mask" : xxx, "input" : xxx } // 模型输入名可通过 Netron 查看模型获知
engine = builder.build_with_name('path/to/jit/module', dummy_inputs)
'''

# 3. build engine with calibration cache
builder = forward.TorchBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8") //（可选）若不设置, 则默认为 FP32
engine = builder.build('path/to/jit/module', dummy_inputs)
# engine = builder.build_with_name('path/to/jit/module', dummy_inputs) 
```

## Customized Calibration cache file

After calibration cache files are generated, they can be modified as customized calibration cache files for special scenarios. The style of calibration cache files are: `Output_Tensor_Name_of_Layer:float_scale_value`, for example:

```txt
TRT-7000-EntropyCalibration
INPUT0:4.6586
(Unnamed Layer* 2) [Activation]_output:10.8675
(Unnamed Layer* 3) [Pooling]_output:11.1173
...
```

### Cpp

```c++
// 1. load calibration cache file
const std::string cacheTableName = "calibrator.cache"; 
const std::string algo = "entropy"; // [entropy | entropy_2 | minmax]
int batch_size = 1;    // batch size
// create TrtInt8Calibrator
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
# 1. load calibration cache file
cacheTableName = "calibrator.cache"
algo = "entropy" # algo = forward.ENTROPY_CALIBRATION | "entropy_2" | "minmax"
batch_size = 1
calib = forward.TrtInt8Calibrator(cacheTableName , algo , batch_size)

customized_cache_file = "path/to/scale_file.txt"
calib.set_scale_file(customized_cache_file)

# 2. build engine
builder.set_calibrator(calibrator)
builder.set_mode("int8") 
engine = builder.build('path/to/jit/module', dummy_inputs)
```