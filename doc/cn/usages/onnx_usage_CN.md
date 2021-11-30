# Forward-ONNX

----

- [Forward-ONNX](#forward-onnx)
  - [环境依赖](#环境依赖)
  - [ONNX 模型导出](#onnx-模型导出)
  - [项目构建](#项目构建)
  - [动态批量输入](#动态批量输入)
    - [CMake 选项](#cmake-选项)
    - [Builder 选项](#builder-选项)
    - [ONNX 支持](#onnx-支持)
  - [Cpp 示例](#cpp-示例)
    - [Cpp INT8 示例](#cpp-int8-示例)
    - [Cpp BERT INT8 示例](#cpp-bert-int8-示例)
  - [Python 示例](#python-示例)
    - [Python INT8 示例](#python-int8-示例)
    - [Python BERT-INT8 示例](#python-bert-int8-示例)
  - [使用手工设定 scale 进行量化](#使用手工设定-scale-进行量化)
    - [Cpp](#cpp)
    - [Python](#python)

----

## 环境依赖

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2)
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.12.2
- GCC >= 5.4.0, ld >= 2.26.1

## ONNX 模型导出

ONNX 模型通常由已训练好的 TensorFlow / PyTorch / Keras 模型转换获得。以下示例演示如何使用 PyTorch 转换 ONNX 模型（默认用户已安装 `torch` 包）。

```python
import torch
import torch.onnx
import torchvision.models as models

# 同时导出 PyTorch JIT 模型，便于验证转换后 ONNX 模型的精度
model = models.resnet50(pretrained=True)
model.cpu()
model.eval()

traced_model = torch.jit.trace(model, inputs)
torch.jit.save(traced_model, 'resnet50.pth')

# 输入输出名字可通过 Netron 查看导出的 PyTorch 模型获得
input_names = ["x"]
output_names = ["495"]

torch.onnx.export(model, inputs, 'resnet50.onnx', verbose=True, input_names=input_names, output_names=output_names)
```

## 项目构建

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

## 动态批量输入

**注意**: 仅当 TensorRT 版本大于 7.1.xx.xx 时，才可在 INT8 推理模式下使用动态批量输入功能。

### CMake 选项

- 当动态批量功能开关打开 `-DENABLE_DYNAMIC_BATCH=ON` 时，Engine 可用 1 < `batch_size` < `max_batch_size` 的任意 batch_size 来进行推理。另外，也可以设置一个 `opt_batch_size`，则 Engine 将会根据此批量大小进行针对性优化。

### Builder 选项

- `max_batch_size`: 构建 Engine 时，`max_batch_size` 将会被设置为 `batch_size` ;
- `opt_batch_size`: 构建 Engine 时，需要显式调用接口来设置 `opt_batch_size` 。如果没有被设置，则 `opt_batch_size` 将与 `max_batch_size` 保持一致。
  - C++ 调用接口: `onnx_builder.SetOptBatchSize(opt_batch_size)`
  - Python 调用接口: `onnx_builder.set_opt_batch_size(opt_batch_size)`

### ONNX 支持

- 若 ONNX 模型是通过 `torch` 包导出的，在调用 `torch.onnx.export` 时，需通过 `dynamic_axes` 指定动态的维度，使用方法可参考 [TORCH.ONNX](https://pytorch.org/docs/stable/onnx.html) 。
- 若使用其他框架导出 ONNX 模型，用户需要保证导出后的 ONNX 模型支持动态输入。
- **注意**：目前 Forward 仅支持 `batch_size` 作为动态输入，不支持多维度的动态输入，且 `batch_size` 必须是模型输入的第一个维度坐标，即输入的维度可以是 `[batch_size, C, H, W]`，但不允许是 `[C, H, W, batch_size]` 等等。

## Cpp 示例

```c++
// 1. 构建 Builder
fwd::OnnxBuilder onnx_builder;

std::string model_path = "path/to/onnx/model";
const std::string infer_mode = "float32";  // float32 / float16 / int8_calib / int8
onnx_builder.SetInferMode(infer_mode);

// 2. 准备输入数据
const auto shape = std::vector<int>{1, 3, 224, 224};  // 伪输入的数据类型和维度须与真实输入保持一致 
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

// 3. 构建 Engine
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);

// 4. 执行推理
std::vector<fwd::Tensor> outputs;
if (!onnx_engine->Forward(inputs, outputs)) {
  std::cerr << "Engine forward error! " << std::endl;
  return -1;
}

// 5. 处理输出数据
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

### Cpp INT8 示例

```c++
#include "common/trt_batch_stream.h"
// 继承实现数据批量获取工具类
class ImgBatchStream : public IBatchStream {
  // 是否还有batch
  bool next() override{...};
  // 获取输入, 一个输入为一个vector<float>
  std::vector<std::vector<float>> getBatch() override{...};
  // 返回一个batch的大小
  int getBatchSize() const override{...};
  // 返回getBatch中每个输入的长度
  std::vector<int64_t> size() const override{...};
} std::shared_ptr<IBatchStream> ibs = std::make_shared<ImgBatchStream>();
// 创建量化器实例，参数分别为BatchStream，缓存名，量化算法名[entropy | entropy_2 | minmax]
std::shared_ptr<TrtInt8Calibrator> calib =
    std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "entropy");

fwd::OnnxEngine onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);
```

### Cpp BERT INT8 示例

- 相对于普通 INT8 的构建, BERT INT8 需要提前用 `int8_calib` 模式生成一个 calibrator 码本, 然后再用 `int8` 模式自动加载生成的码本来构建 int8 推理引擎. 

```c++
#include "common/trt_batch_stream.h"
// 继承实现数据批量获取工具类，可参考 test_batch_stream.h 里面的 BertStream
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
// 创建量化器实例，参数分别为BatchStream，缓存名，量化算法名用 minmax
std::shared_ptr<TrtInt8Calibrator> calib =
    std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "minmax");

// 构建码本
fwd::OnnxBuilder onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8_calib");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);

// 使用码本构建推理引擎
fwd::OnnxBuilder onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);
```

## Python 示例

构建成功后，须要将 `build/bin` 目录下的 `forward*.so` (Linux) 或 `forward*.pyd` (Windows) 拷贝到 Python 工作目录下。

``` python
import forward
import numpy as np

# 1. 构建 Engine
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

# 2. 执行推理
inputs = np.random.rand(1, 3, 224, 224).astype('float32')
outputs = engine.forward([inputs])
```

### Python INT8 示例

```python
import forward
import numpy as np


# 1. 继承实现数据提供工具类
class MBatchStream(forward.IPyBatchStream):
    def __init__(self):
        forward.IPyBatchStream.__init__(self)  # 必须调用父类的初始化方法
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

# 2. 构建 Engine
builder.set_mode("int8")
engine = builder.build('path/to/onnx/model')
```

### Python BERT-INT8 示例

- 相对于普通 INT8 的构建，BERT INT8 需要提前用 `int8_calib` 模式生成一个 calibrator 码本，然后再用 `int8` 模式自动加载生成的码本来构建 int8 推理引擎。

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

# 2. 构建码本
builder = forward.OnnxBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8_calib")  # （可选）若不设置，则默认为 FP32
engine = builder.build('path/to/onnx/model')

# 3. 使用码本构建引擎
builder = forward.OnnxBuilder()
builder.set_calibrator(calibrator)
builder.set_mode("int8")  # （可选）若不设置，则默认为 FP32
engine = builder.build('path/to/onnx/model')
```

## 使用手工设定 scale 进行量化

使用 scale 量化不需要实现 IBatchStream 工具类，但需要提供每层的 scale

scale 文件可由非手动量化的缓存文件修改得：

格式为每行为需要设定的层输出名：缩放 scale 值 `TensorName: float_scale`

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
const std::string cacheTableName = "calibrator.cache";  // calib缓存文件名
const std::string algo = "entropy";  // 量化算法名[entropy | entropy_2 | minmax]
int batch_size = 1;                  // batch size
// 创建量化器实例
std::shared_ptr<TrtInt8Calibrator> calib =
    std::make_shared<TrtInt8Calibrator>(cacheTableName, algo, batch_size);
const std::string customized_cache_file = "path/to/scale_file.txt";
calib->setScaleFile(customized_cache_file);

// 2. build engine
fwd::OnnxBuilder onnx_builder;
onnx_builder.SetCalibrator(calib);
onnx_builder.SetInferMode("int8");
std::shared_ptr<fwd::OnnxEngine> onnx_engine = onnx_builder.Build(model_path);
```

### Python

```python
# 1. 准备量化器
cacheTableName = "calibrator.cache"
algo = "entropy"  # algo = forward.ENTROPY_CALIBRATION | "entropy_2" | "minmax"
batch_size = 1
calib = forward.TrtInt8Calibrator(cacheTableName, algo, batch_size)

customized_cache_file = "path/to/scale_file.txt"
calib.set_scale_file(customized_cache_file)

# 2. 构建 engine
builder.set_calibrator(calibrator)
builder.set_mode("int8")
engine = builder.build('path/to/onnx/model')
```
