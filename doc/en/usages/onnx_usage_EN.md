# Forward-ONNX

----

- [Forward-ONNX](#forward-onnx)
  - [Prerequisites](#prerequisites)
  - [Export ONNX Models](#export-onnx-models)
  - [Build](#build)
  - [Dynamic Batch Inputs](#dynamic-batch-inputs)
    - [CMake Options](#cmake-options)
    - [Builder Options](#builder-options)
    - [ONNX Support](#onnx-support)
  - [Cpp Example](#cpp-example)
    - [Cpp INT8 Example](#cpp-int8-example)
    - [Cpp BERT INT8 Example](#cpp-bert-int8-example)
  - [Python Example](#python-example)
    - [Python INT8 Example](#python-int8-example)
    - [Python BERT INT8 Example](#python-bert-int8-example)
  - [Customized Calibration Cache File](#customized-calibration-cache-file)
    - [Cpp](#cpp)
    - [Python](#python)

----

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2)
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1

## Export ONNX Models

ONNX models are usually exported from trained TensorFlow / PyTorch / Keras models. The following example demonstrates how to use PyTorch to convert ONNX models (the `torch` package is installed by default).

```python
import torch
import torch.onnx
import torchvision.models as models


# export the PyTorch JIT model to facilitate verification of the accuracy of the converted ONNX model
model = models.resnet50(pretrained=True)
model.cpu()
model.eval()

traced_model = torch.jit.trace(model, inputs)
torch.jit.save(traced_model, 'resnet50.pth')

# input and output names can be obtained by viewing the exported PyTorch model through Netron
input_names = ["x"]
output_names = ["495"]

torch.onnx.export(model, inputs, 'resnet50.onnx', verbose=True, input_names=input_names, output_names=output_names)
```

## Build

```bash
mkdir build
cd build
cmake ..                               \ 
-DTensorRT_ROOT="path/to/TensorRT"     \ 
-DENABLE_TENSORFLOW=OFF                \ 
-DENABLE_TORCH=OFF                     \ 
-DENABLE_KERAS=OFF                     \ 
-DENABLE_ONNX=ON                       \ 
-DENABLE_UNIT_TESTS=ON                 \ 
-DBUILD_PYTHON_LIB=OFF                 \ 
-DPYTHON_EXECUTABLE="/path/to/python3"

make -j
```

## Dynamic Batch Inputs

**Notice**: Dynamic batch inputs support INT8 mode ONLY when TensorRT version > 7.1.xx.xx.

### CMake Options

- Set `-DENABLE_DYNAMIC_BATCH=ON` to support dynamic batch inputs, then `batch_size` between 1 and `max_batch_size` can be valid during inference periods. In this case, Forward engines will be optimized according to `opt_batch_size`.

### Builder Options

- `max_batch_size`: When building engines, `max_batch_size` is set as `batch_size` for Forward engines;
- `opt_batch_size`: When building engines, `opt_batch_size` should be explicitly set by following interfaces, otherwise, it will be set as the same as `max_batch_size`.
  - Cpp interface: `onnx_builder.SetOptBatchSize(opt_batch_size)`
  - Python interface: `onnx_builder.set_opt_batch_size(opt_batch_size)`

### ONNX Support

- If the ONNX model is exported through the `torch` package, when calling `torch.onnx.export`, the dynamic dimensions need to be specified by `dynamic_axes`, the usage can refer to [TORCH.ONNX](https://pytorch.org/docs/stable/onnx.html).
- If other frameworks are used to export ONNX models, users need to ensure that the exported ONNX models support dynamic batch inputs.
- **Note**: Currently Forward only supports `batch_size` to be dynamic, and does not support multi-dimensional dynamic inputs, and `batch_size` must be the first dimension of the model inputs, that is, the input dimension can be `[batch_size , C, H, W]`, but `[C, H, W, batch_size]` and so on are not allowed.

## Cpp Example

```c++
// 1. build Builder
fwd::OnnxBuilder onnx_builder;

std::string model_path = "path/to/onnx/model";
const std::string infer_mode = "float32";  // float32 / float16 / int8_calib / int8
onnx_builder.SetInferMode(infer_mode);

// 2. prepare inputs
// the data type and dimension of the pseudo inputs must be consistent with the real inputs
const auto shape = std::vector<int>{1, 3, 224, 224};
const auto volume = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
std::vector<float> data;
data.resize(volume);
std::memset(data.data(), 0, sizeof(float) * volume);

fwd::Tensor input;
input.data = data.data();
input.dims = shape;
input.data_type = fwd::DataType::FLOAT;
input.device_type = fwd::DeviceType::CPU;
const std::vector<fwd::Tensor> inputs{input};

// 3. build Engine
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);

// 4. do inference
std::vector<fwd::Tensor> outputs;
if (!onnx_engine->Forward(inputs, outputs)) {
  std::cerr << "Engine forward error! " << std::endl;
  return -1;
}

// 5. process outputs
std::vector<std::vector<float>> h_outputs;
for (size_t i = 0; i < outputs.size(); ++i) {
  std::vector<float> h_out;
  const auto &out_shape = outputs[i].dims;
  auto out_volume =
      std::accumulate(out_shape.cbegin(), out_shape.cend(), 1, std::multiplies<int>());
  h_out.resize(out_volume);
  MemcpyDeviceToHost(h_out.data(), reinterpret_cast<float *>(outputs[i].data), out_volume);
  h_outputs.push_back(std::move(h_out));
}
```

### Cpp INT8 Example

```c++
#include "common/trt_batch_stream.h"

// inherit from IBatchStream and implement override functions
class ImgBatchStream : public IBatchStream {
  // check if has next batch
  bool next() override{...};
  // get next batch inputs
  std::vector<std::vector<float>> getBatch() override{...};
  // get batch size of next batch inputs
  int getBatchSize() const override{...};
  // get volume size of next batch inputs
  std::vector<int64_t> size() const override{...};
} std::shared_ptr<IBatchStream> ibs = std::make_shared<ImgBatchStream>();
// create TrtInt8Calibrator, algorithm can be [entropy | entropy_2 | minmax]
std::shared_ptr<TrtInt8Calibrator> calib =
    std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "entropy");

fwd::OnnxEngine onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);
```

### Cpp BERT INT8 Example

- Unlike building INT8 engines for normal models, building INT8 engines for BERT models has to use `int8_calib` mode to generate a calibration cache as CodeBook at first, and then use `int8` mode to build engines with the calibration cache file.

```c++
#include "common/trt_batch_stream.h"
// inherit from IBatchStream
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
  std::vector<std::vector<int>> mData;  // input_ids, attention_mask, segment_ids
};

std::shared_ptr<IBatchStream> ibs = std::make_shared<BertBatchStream>();
// use 'minmax' algorithm
std::shared_ptr<TrtInt8Calibrator> calib =
    std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "minmax");

// build with int8_calib mode to generate a calibration cache file
fwd::OnnxBuilder onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8_calib");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);

// build int8 engine with the saved calibration cache file
fwd::OnnxBuilder onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);
```

## Python Example

After building Forward project, `forward*.so (Linux)` or `forward*.pyd` (Windows) in `build/bin` directory should be copied into WORK_DIR of Python project.

``` python
import forward
import numpy as np


# 1. build Engine
builder = forward.OnnxBuilder()

infer_mode = 'float32'  # float32 / float16 / int8_calib / int8
builder.set_mode(infer_mode)

model_path = 'path/to/onnx/model'
engine = builder.build(model_path)

need_save = True
if need_save:
    engine_path = model_path + ".engine"
    engine.save(engine_path)

    engine = forward.OnnxEngine()
    engine.load(engine_path)

# 2. do inference
inputs = np.random.rand(1, 3, 224, 224).astype('float32')
outputs = engine.forward([inputs])
```

### Python INT8 Example

```python
import forward
import numpy as np


# 1. inherit forward.IPyBatchStream
class MBatchStream(forward.IPyBatchStream):
    def __init__(self):
        forward.IPyBatchStream.__init__(self)  # required
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
        return [1 * 24 * 24 * 3]

    def getNumpyBatch(self):
        return [np.random.randn(1 * 24 * 24 * 3)]


bs = MBatchStream()
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.MINMAX_CALIBRATION)

builder = forward.OnnxBuilder()
builder.set_calibrator(calibrator)

# 2. build engine
builder.set_mode("int8")
engine = builder.build('path/to/onnx/model')
```

### Python BERT INT8 Example

- Unlike building INT8 engines for normal models, building INT8 engines for BERT models has to use `int8_calib` mode to generate a calibration cache as CodeBook at first, and then use `int8` mode to build engines with the calibration cache file.

```python
import forward
import bert_helpers.tokenization
import bert_helpers.data_preprocessing as dp
import numpy as np


# 1. inherit from forward.IPyBatchStream
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
        # self.device_inputs = [cuda.mem_alloc(self.max_seq_length * trt.int32.itemsize * self.batch_size) for binding in range(3)]

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
                example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride,
                self.max_query_length)
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

# 2. generate calibration cache as CodeBook
builder = forward.OnnxBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8_calib")  # (optional) if not set, the default is FP32
engine = builder.build('path/to/onnx/model')

# 3. build Engine with calibration cache
builder = forward.OnnxBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8")  # (optional) if not set, the default is FP32
engine = builder.build('path/to/onnx/model')
```

## Customized Calibration Cache File

After calibration cache files are generated, they can be modified as customized calibration cache files for special scenarios. The style of calibration cache files are: `Output_Tensor_Name_of_Layer:float_scale_value`, for example:

```
TRT-7000-EntropyCalibration
INPUT0:4.6586
(Unnamed Layer* 2) [Activation]_output:10.8675
(Unnamed Layer* 3) [Pooling]_output:11.1173
...
```

### Cpp

```c++
// 1. load calibration cache file
const std::string cacheTableName = "calibrator.cache";  // calib cache file name
const std::string algo = "entropy";  // [entropy | entropy_2 | minmax]
int batch_size = 1;                  // batch size
// create TrtInt8Calibrator
std::shared_ptr<TrtInt8Calibrator> calib =
    std::make_shared<TrtInt8Calibrator>(cacheTableName, algo, batch_size);
const std::string customized_cache_file = "path/to/scale_file.txt";
calib->setScaleFile(customized_cache_file);

// 2. build Engine
fwd::OnnxBuilder onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);
```

### Python

```python
# 1. load calibration cache file
cacheTableName = "calibrator.cache"
algo = "entropy"  # algo = forward.ENTROPY_CALIBRATION | "entropy_2" | "minmax"
batch_size = 1
calib = forward.TrtInt8Calibrator(cacheTableName, algo, batch_size)

customized_cache_file = "path/to/scale_file.txt"
calib.set_scale_file(customized_cache_file)

# 2. build Engine
builder.set_calibrator(calibrator)
builder.set_mode("int8")
engine = builder.build('path/to/onnx/model')
```
