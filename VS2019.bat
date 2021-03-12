if not exist build ( md build )

cd build

REM cmake .. -G "Visual Studio 16 2019" -A x64

cmake .. -G "Visual Studio 16 2019" -A x64 ^
-DTensorRT_ROOT=path\to\TensorRT ^
-DCMAKE_PREFIX_PATH="path\to\libtorch\share\cmake;path\to\HDF5\1.12.0\cmake\hdf5" ^
-DENABLE_TENSORFLOW=ON ^
-DENABLE_TORCH=ON ^
-DENABLE_KERAS=OFF ^
-DENABLE_INFER_TESTS=OFF ^
-DENABLE_RNN=ON

cd ..
