rm -rf build
mkdir build

ENABLE_TORCH=ON
ENABLE_TENSORFLOW=ON

TensorRT_ROOT=/path/to/TensorRT
LibTorch=/path/to/libtorch
LibTensorflow=/path/to/tensorflow

cd build
make clean
cmake .. -DENABLE_TORCH=$ENABLE_TORCH -DENABLE_TENSORFLOW=$ENABLE_TENSORFLOW -DTensorRT_ROOT=$TensorRT_ROOT -DCMAKE_PREFIX_PATH=$LibTorch -DTensorflow_ROOT=$LibTensorflow

make -j
