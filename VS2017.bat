if not exist build ( md build )

cd build

REM cmake .. -G "Visual Studio 15 2017" -A x64

cmake .. -G "Visual Studio 15 2017" -A x64 ^
-DTensorRT_ROOT="path/to/TensorRT" ^
-DCMAKE_PREFIX_PATH="path/to/LibTorch" ^
-DENABLE_TENSORFLOW=OFF ^
-DENABLE_TORCH=OFF ^
-DENABLE_KERAS=OFF ^
-DENABLE_ONNX=OFF ^
-DENABLE_INFER_TESTS=OFF ^
-DENABLE_RNN=OFF ^
-DENABLE_UNIT_TESTS=OFF

cd ..
