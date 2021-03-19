# Forward-TensorFlow

----

- [Forward-TensorFlow](#forward-tensorflow)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Dynamic batch inputs](#dynamic-batch-inputs)
    - [CMake options](#cmake-options)
    - [Setting](#setting)
  - [C++ Example](#cpp-example)
    - [C++ INT8 Example](#cpp-int8-example)
    - [C++ BERT INT8 Example](#cpp-bert-int8-example)
  - [Python Example](#python-example)
    - [Python INT8 Example](#python-int8-example)
    - [Python BERT-INT8 Example](#python-bert-int8-example)

----

## Prerequisites

- NVIDIA CUDA >= 10.0, CuDNN >= 7 (Recommended version: CUDA 10.2 )
- TensorRT >= 7.0.0.11,  (Recommended version: TensorRT-7.2.1.6)
- CMake >= 3.10.1
- GCC >= 5.4.0, ld >= 2.26.1
- (Tensorflow) TensorFlow == 1.15.0 (download [Tensorflow 1.15.0](https://github.com/neargye-forks/tensorflow/releases) and unzip it to `source/third_party/tensorflow/lib`)

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
-DENABLE_TENSORFLOW=ON \
-DENABLE_KERAS=OFF

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
fwd::TfBuilder tf_builder;

std::string model_path = "path/to/tf.pb";
const std::string infer_mode = "float32"; // float32 / float16 / int8_calib /int8
const int batch_size = 32;

// 伪输入与真实输入的维度和数据类型须要保持一致
std::shared_ptr<TF_Tensor> dummy_input = fwd::tf_::CreateRandomTensor<float>(
      TF_FLOAT, {batch_size, 12, 24, 3});
std::unordered_map<std::string, TF_Tensor*> dummy_input_map;
dummy_input_map.insert({"input_name", dummy_input});

tf_builder.SetInferMode(infer_mode); // （可选）若不设置, 则默认为 FP32
std::shared_ptr<fwd::TfEngine> tf_engine = tf_builder.Build(model_path, dummy_input_map );

bool need_save = true;
if (!need_save){
    std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> outputs = tf_engine->ForwardWithName(dummy_input_map);
}else{
    // save engine
    std::string engine_file = "path/to/out/engine";
    tf_engine->Save(engine_file);
    // load saved engine
    fwd::TfEngine tf_engine;
    tf_engine.Load(engine_file);

    std::unordered_map<std::string, TF_Tensor*> input_map = dummy_input_map;  
    std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> outputs = tf_engine.ForwardWithName(dummy_input_map);
}

```

### Cpp INT8 Example

```c++
#include "common/trt_batch_stream.h"
// 1. 创建 INT 8 量化器
// 继承实现数据批量获取工具类
class TestBatchStream : public IBatchStream {
    // 是否还有batch
    bool next() override {...};
    // 获取输入, 一个输入为一个vector<float>
    std::vector<std::vector<float>> getBatch() override {...};
    // 返回一个batch的大小
    int getBatchSize() const override {...};
    // 返回getBatch中每个输入的长度
    std::vector<int64_t> size() const override {...};
}
std::shared_ptr<IBatchStream> ibs = std::make_shared<TestBatchStream>();
// 创建量化器实例, 参数分别为BatchStream, 缓存名, 量化算法名[entropy | entropy_2 | minmax]
std::shared_ptr<TrtInt8Calibrator> calib = std::make_shared<TrtInt8Calibrator>(ibs, "calibrator.cache", "entropy");

// 2. 构建 Engine
fwd::TfBuilder tf_builder;

std::string model_path = "path/to/tf.pb";
const std::string infer_mode = "float32"; // or float16
const int batch_size = 32;

// 伪输入与真实输入的维度和数据类型须要保持一致
std::unordered_map<std::string, TF_Tensor*> dummy_input_map = xxx;
tf_builder.SetCalibrator(calib);

tf_builder.SetInferMode("int8"); 
std::shared_ptr<fwd::TfEngine> tf_engine = tf_builder.Build(model_path, dummy_inputs);
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
fwd::TfBuilder tf_builder;
tf_builder.SetCalibrator(calib);

// 构建码本
tf_builder.SetInferMode("int8_calib"); 
std::shared_ptr<fwd::TfEngine> tf_engine = tf_builder.Build(model_path, dummy_inputs);

// 使用码本构建推理引擎
fwd::TfBuilder tf_builder;
tf_builder.SetCalibrator(calib);
tf_builder.SetInferMode("int8");
std::shared_ptr<fwd::TfEngine> tf_engine = tf_builder.Build(model_path, dummy_inputs);
```

## Python Example

构建成功后,  需要将 `build/bin` 目录下的 `forward*.so`(Linux) or `forward*.pyd`(Windows) 拷贝到 Python 工作目录下.

``` python
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
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.MINMAX_CALIBRATION)

builder = forward.TfBuilder()
batch_size = 16
dummy_input = {"input1" : np.ones([batch_size , 24, 24, 3], dtype='float32'), 
               "input2" : np.ones([batch_size , 24, 24, 3], dtype='float32'),
               "input3" : np.ones([batch_size , 24, 24, 3], dtype='float32')}

builder.setCalibrator(calibrator)

# 2. 构建 engine
builder.set_mode("int8") //（可选）若不设置, 则默认为 FP32
tf_engine = builder.build('path/to/model', dummy_inputs)

need_save = True
if need_save:
    # save engine
    engine_path = 'path/to/out/engine'
    tf_engine.save(engine_path)

    # load engine
    tf_engine = forward.TfEngine()
    tf_engine.load(engine_path)

inputs = dummy_input
outputs = tf_engine.forward(inputs) # dict_type outputs
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
calibrator = forward.TrtInt8Calibrator(bs, "calibrator.cache", forward.MINMAX_ENTROPY)

# 2. 构建码本
builder = forward.TfBuilder()
builder.setCalibrator(calibrator)
builder.set_mode("int8_calib") 
engine = builder.build('path/to/jit/module', dummy_inputs)

# 3. 使用码本构建 Engine
builder = forward.TfBuilder()
builder.setCalibrator(calibrator)
builder.set_mode("int8") 
engine = builder.build('path/to/jit/module', dummy_inputs)
```
