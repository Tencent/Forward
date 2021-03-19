if [ ! -d "build" ]
then
mkdir build
fi

cd build
make clean
cmake .. -DTensorRT_ROOT=/path/to/TensorRT-7.2.1.6 -DCMAKE_PREFIX_PATH=/path/to/libtorch

make -j

