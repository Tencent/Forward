if [ ! -e "./source/third_party/tensorflow/lib/libtensorflow.so" ]
then
echo "tensorflow lib not found. installing..."
mkdir tensorflow_lib
curl -L https://github.com/Neargye/tensorflow/releases/download/v1.15.0/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz -o tensorflow.tar.gz 
tar -xzf tensorflow.tar.gz --directory tensorflow_lib 
cp -rv tensorflow_lib/lib/ source/third_party/tensorflow/ 
rm -rv tensorflow_lib tensorflow.tar.gz 
echo "tensorflow lib installed."
fi

if [ ! -d "build" ]
then
mkdir build
fi

cd build
make clean
cmake .. -DTensorRT_ROOT="path/to/TensorRT"     \ 
         -DCMAKE_PREFIX_PATH="path/to/LibTorch" \ 
         -DENABLE_TENSORFLOW=OFF                \ 
         -DENABLE_TORCH=OFF                     \ 
         -DENABLE_KERAS=OFF                     \ 
         -DENABLE_ONNX=OFF                      \ 
         -DBUILD_PYTHON_LIB=OFF                 \ 
         -DENABLE_RNN=OFF                       \ 
         -DENABLE_INFER_TESTS=OFF               \ 
         -DENABLE_UNIT_TESTS=OFF                \ 
         -DPYTHON_EXECUTABLE=$(which python3)

make -j
