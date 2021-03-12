if not exist build ( md build )

cd build

cmake .. -G "Visual Studio 16 2019" -A x64 ^
-DENABLE_PROFILING=ON ^
-DBUILD_PYTHON_LIB=OFF ^
-DTensorRT_ROOT=path\to\TensorRT ^
-DTORCH_CMAKE_PATH=path\to\libtorch\share\cmake

cd ..
