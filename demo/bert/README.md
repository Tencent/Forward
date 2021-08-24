# Demo for building BERT model

**Notice**: The name of INPUT in models can be viewed by model viewers, such as [Netron](https://github.com/lutzroeder/Netron).

## Tensorflow-BERT
0. Download BERT-model from [BERT](https://github.com/google-research/bert/blob/master/README.md). BERT-tiny is downloaded as example.

1. Utilize the script [export_bert.py](../../python/bert_helpers/export_bert.py) to export `frozen_bert.pb`.

2. Build Forward libraries refer to [CMake-Build](../../doc/en/usages/cmake_build_EN.md). `-DENABLE_TENSORFLOW=ON`, `-DBUILD_PYTHON_LIB=ON` and `-DPYTHON_EXECUTABLE=$(which python3)` should be specified.

3. Copy Forward-Python library to this directory.

4. Run the python script `test_tf_bert.py`

## Torch-BERT
0. Install `transformers` and `torch`.

1. Build Forward libraries refer to [CMake-Build](../../doc/en/usages/cmake_build_EN.md). `-DENABLE_TORCH=ON`, `-DBUILD_PYTHON_LIB=ON` and `-DPYTHON_EXECUTABLE=$(which python3)` should be specified.

2. Copy Forward-Python library to this directory.

3. Run the python script `test_torch_bert.py`

## ONNX-BERT
0. Install `transformers` and convert an ONNX model using the `transformers.onnx` package. The checkpoint `bert-base-uncased` is exported as `python -m transformers.onnx --model=bert-base-uncased bert.onnx`.

1. Build Forward libraries refer to [CMake-Build](../../doc/en/usages/cmake_build_EN.md). `-DENABLE_ONNX=ON`, `-DBUILD_PYTHON_LIB=ON` and `-DPYTHON_EXECUTABLE=$(which python3)` should be specified.

2. Copy Forward-Python library to this directory.

3. Run the python script `test_onnx_bert.py`
