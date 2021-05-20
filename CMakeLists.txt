# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
#
# ╔════════════════════════════════════════════════════════════════════════════════════════╗
# ║──█████████╗───███████╗───████████╗───██╗──────██╗───███████╗───████████╗───████████╗───║
# ║──██╔══════╝──██╔════██╗──██╔════██╗──██║──────██║──██╔════██╗──██╔════██╗──██╔════██╗──║
# ║──████████╗───██║────██║──████████╔╝──██║──█╗──██║──█████████║──████████╔╝──██║────██║──║
# ║──██╔═════╝───██║────██║──██╔════██╗──██║█████╗██║──██╔════██║──██╔════██╗──██║────██║──║
# ║──██║─────────╚███████╔╝──██║────██║──╚████╔████╔╝──██║────██║──██║────██║──████████╔╝──║
# ║──╚═╝──────────╚══════╝───╚═╝────╚═╝───╚═══╝╚═══╝───╚═╝────╚═╝──╚═╝────╚═╝──╚═══════╝───║
# ╚════════════════════════════════════════════════════════════════════════════════════════╝
#

cmake_minimum_required(VERSION 3.12.2 FATAL_ERROR)

project(Forward LANGUAGES C CXX CUDA)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 OLD)
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type")
endif()

set(CMAKE_CXX_STANDARD 11)

include(CMakeDependentOption)

# Enable TensorRT
option(ENABLE_TENSORRT "Enable TensorRT" ON)
# Enable Torch
option(ENABLE_TORCH "Enable Torch" OFF)
# Enable TensorFlow
option(ENABLE_TENSORFLOW "Enable TensorFlow" OFF)
# Enable Keras
option(ENABLE_KERAS "Enable Keras" OFF)
# Enable profiling
option(ENABLE_PROFILING "Enable profiling" OFF)
# Enable logging
option(ENABLE_LOGGING "Enable logging" OFF)
# Enable dynamic batch size
option(ENABLE_DYNAMIC_BATCH "Enable dynamic batch size" OFF)
# Build Python Lib
option(BUILD_PYTHON_LIB "Build Python Lib" OFF)
# Enable Inference Tests (need OpenCV)
option(ENABLE_INFER_TESTS "Enable Inference Tests" OFF)
# Enable RNN models forward
option(ENABLE_RNN "Enable RNN models forward" OFF)
# Enable unit tests
option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)

if(MSVC)
  set(USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(BUILD_SHARED_LIBS OFF)

  if (BUILD_PYTHON_LIB) 
    message(FATAL_ERROR "BUILD_PYTHON_LIB=ON is NOT SUPPORTED on WINDOWS platform.")
  endif()
else(MSVC)
  set(BUILD_SHARED_LIBS ON)
endif(MSVC)

# Options CMAKE_DEPENDENT_OPTION(ENABLE_UNIT_TESTS "Enable unit tests" OFF
# "BUILD_PYTHON_LIB" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CMake path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

# Cuda
find_package(CUDA REQUIRED)
if (${CUDA_VERSION_MAJOR} LESS 10)
  message(FATAL_ERROR "Forward requires CUDA_VERSION_MAJOR >= 10, but the given is ${CUDA_VERSION_MAJOR}")
endif()
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -use_fast_math \
    -Xfatbin \
    -compress-all \
    -Xcompiler=-Wno-deprecated-declarations \
")

set(GPU_ARCHS
    "60;61;70;75"
    CACHE STRING "GPU SM")
foreach(arch ${GPU_ARCHS})
  set(CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
list(GET GPU_ARCHS -1 ptx)
set(CUDA_NVCC_FLAGS
    "${CUDA_NVCC_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")

message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})

# Enable compiler warnings
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /WX-")
else()
  # add_compile_options(-std=c++11)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -fPIC -std=c++11 -O3")
  set(CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-declarations -Xcompiler -fPIC -O3"
      CACHE STRING "cuda flags")
endif()

# support RNN models
if(ENABLE_RNN)
  add_definitions(-DSUPPORT_RNN=1)
endif(ENABLE_RNN)

# Dynamic Batch
if(ENABLE_DYNAMIC_BATCH)
  add_definitions(-DUSE_DYNAMIC_BATCH=1)
endif()

# _GLIBCXX_USE_CXX11_ABI=0 only for pytorch
if (BUILD_PYTHON_LIB AND ENABLE_TORCH)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message(WARNING "_GLIBCXX_USE_CXX11_ABIT=0 is set for PyTorch libraries. Check dependencies for this flag.")
endif()

# easylogging
if(NOT ENABLE_LOGGING)
  add_definitions(-DELPP_DISABLE_INFO_LOGS=1)
endif()
add_definitions(-DELPP_THREAD_SAFE=1)
include_directories(${PROJECT_SOURCE_DIR}/source/third_party/easyloggingpp)
# add_subdirectory(${PROJECT_SOURCE_DIR}/source/third_party/easyloggingpp)

# Profiling
if(ENABLE_PROFILING)
  add_definitions(-DTRT_INFER_ENABLE_PROFILING=1)
else()
  add_definitions(-DTRT_INFER_ENABLE_PROFILING=0)
endif() # ENABLE_PROFILING

# simple-utils
include_directories(${PROJECT_SOURCE_DIR}/source/third_party/simple-utils)
add_subdirectory(${PROJECT_SOURCE_DIR}/source/third_party/simple-utils)
set_target_properties(simple-utils PROPERTIES FOLDER third_party)

# Cub-1.8.0
include_directories(${PROJECT_SOURCE_DIR}/source/third_party/cub-1.8.0)

# TensorRT
find_package(TensorRT)

if(NOT TensorRT_FOUND) # use sources
  message(
    FATAL_ERROR
      "Can NOT find TensorRT, you may set TensorRT root via TensorRT_ROOT.
            like -DTensorRT_ROOT=D:/Libs/TensorRT-6.0.1.5/")
endif()

include_directories(${TensorRT_INCLUDE_DIRS})
if(MSVC)
  configure_file(${TensorRT_ROOT}/lib/nvinfer.dll
                 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/nvinfer.dll COPYONLY)
  configure_file(${TensorRT_ROOT}/lib/nvinfer.dll
                 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/nvinfer.dll COPYONLY)
  configure_file(${TensorRT_ROOT}/lib/nvinfer_plugin.dll
                 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/nvinfer_plugin.dll COPYONLY)
  configure_file(
    ${TensorRT_ROOT}/lib/nvinfer_plugin.dll
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/nvinfer_plugin.dll COPYONLY)
  configure_file(${TensorRT_ROOT}/lib/myelin64_1.dll
                 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/myelin64_1.dll COPYONLY)
  configure_file(
    ${TensorRT_ROOT}/lib/myelin64_1.dll
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/myelin64_1.dll COPYONLY)
endif(MSVC)

# common
include_directories(${PROJECT_SOURCE_DIR}/source)

# TensorRT
if(ENABLE_TENSORRT)
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/trt_engine)
endif(ENABLE_TENSORRT)

# Torch
if(ENABLE_TORCH)
  if(BUILD_PYTHON_LIB)
    if(NOT PYTHON_EXECUTABLE)
      message(STATUS "Do not detected PYTHON_EXECUTABLE, set it to python3")
      set(PYTHON_EXECUTABLE python3)
    endif(NOT PYTHON_EXECUTABLE)
    execute_process(
      COMMAND
        ${PYTHON_EXECUTABLE} -c
        "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
      OUTPUT_VARIABLE TorchPath)
    list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
  else()
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
  endif()

  find_package(Torch)
  if(NOT TORCH_FOUND)
    message(
      FATAL_ERROR
        "Please set libtorch/share/cmake/ path to TORCH_CMAKE_PATH,
            like -DTORCH_CMAKE_PATH=D:/libtorch/share/cmake/Torch
            or -DTORCH_CMAKE_PATH=/usr/local/share/cmake")
  endif()
  if(MSVC)
    configure_file(${TORCH_INSTALL_PREFIX}/lib/torch.dll
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/torch.dll COPYONLY)
    configure_file(${TORCH_INSTALL_PREFIX}/lib/torch.dll
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/torch.dll COPYONLY)
    configure_file(${TORCH_INSTALL_PREFIX}/lib/c10.dll
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/c10.dll COPYONLY)
    configure_file(${TORCH_INSTALL_PREFIX}/lib/c10.dll
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/c10.dll COPYONLY)
  endif(MSVC)

  message(STATUS "Find Torch VERSION: ${Torch_VERSION}")
  add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
  add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
  add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})
  if (${Torch_VERSION} VERSION_GREATER_EQUAL "1.7.0")
      set(CMAKE_CXX_STANDARD 14)
      set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()

  include_directories(${TORCH_INCLUDE_DIRS})
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/fwd_torch)
  add_definitions(-DENABLE_TORCH=1)
endif() # ENABLE_TORCH

# TensorFlow
if(ENABLE_TENSORFLOW)
  set(TENSORFLOW_LIB_DIR "${PROJECT_SOURCE_DIR}/source/third_party/tensorflow")

  if(MSVC)
    # add_compile_options(/W4 /WX) if(NOT CMAKE_CXX_COMPILER_VERSION
    # VERSION_LESS 19.11) add_compile_options(/permissive-) endif()
    configure_file(${TENSORFLOW_LIB_DIR}/lib/tensorflow.dll
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tensorflow.dll COPYONLY)
    configure_file(
      ${TENSORFLOW_LIB_DIR}/lib/tensorflow.dll
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/tensorflow.dll COPYONLY)
  endif()

  include_directories(${TENSORFLOW_LIB_DIR}/include)

  add_subdirectory(${PROJECT_SOURCE_DIR}/source/fwd_tf)

  message("LINK_DIR = ${PROJECT_SOURCE_DIR}/source/third_party/tensorflow/lib")
  link_directories(${PROJECT_SOURCE_DIR}/source/third_party/tensorflow/lib)

  include_directories(${TENSORFLOW_LIB_DIR}/include)

  add_definitions(-DENABLE_TENSORFLOW=1)
  if (BUILD_PYTHON_LIB)
    if(NOT PYTHON_EXECUTABLE)
      message(STATUS "Do not detected PYTHON_EXECUTABLE, set it to python3")
      set(PYTHON_EXECUTABLE python3)
    endif(NOT PYTHON_EXECUTABLE)
    execute_process(
      COMMAND
        ${PYTHON_EXECUTABLE} -c
        "import tensorflow; import os; print(os.path.dirname(tensorflow.pywrap_tensorflow.__file__), end='')"
      OUTPUT_VARIABLE TF_PYLIB_DIR)
    execute_process(
      COMMAND
        ${PYTHON_EXECUTABLE} -c
        "import tensorflow; import os; print(os.path.dirname(tensorflow.python.pywrap_tensorflow.__file__), end='')"
      OUTPUT_VARIABLE TF2_PYLIB_DIR)
    
    message(STATUS "TF_PYLIB_DIR = ${TF_PYLIB_DIR} ${TF2_PYLIB_DIR}")
    find_library(PY_TENSORFLOW_LIB NAMES _pywrap_tensorflow_internal.so PATHS "${TF_PYLIB_DIR}" "${TF2_PYLIB_DIR}")
    if (NOT PY_TENSORFLOW_LIB)
      message(WARNING "_pywrap_tensorflow_internal.so NOT-FOUND in the python-packages! Use libtensorflow.so under the source/third_party/tensorflow/lib. In this case, 'import forward' cannot be used with 'import tensorflow' in the same time.")
      set(PY_TENSORFLOW_LIB tensorflow)
    endif()
  endif()
endif()

if(ENABLE_KERAS)
  if(ENABLE_UNIT_TESTS AND NOT ENABLE_TENSORFLOW)
    message(FATAL_ERROR "Unit test Keras must ENABLE_TENSORFLOW")
  endif()
  # json
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/third_party/json)
  include_directories(
    ${PROJECT_SOURCE_DIR}/source/third_party/json/single_include)
  # hdf5
  # find_package(HDF5)
  if (HDF5_FOUND)
    message("Found HDF5 on system HDF5_INCLUDE_DIRS: " ${HDF5_INCLUDE_DIRS})
    include_directories(${HDF5_INCLUDE_DIRS})
  else()
    message("Use HDF5 on third_party: ${PROJECT_SOURCE_DIR}/source/third_party/hdf5")
    include_directories(${PROJECT_SOURCE_DIR}/source/third_party/hdf5/src)
    include_directories(${PROJECT_SOURCE_DIR}/source/third_party/hdf5/include)
    add_subdirectory(${PROJECT_SOURCE_DIR}/source/third_party/hdf5)
  endif()

  add_subdirectory(${PROJECT_SOURCE_DIR}/source/fwd_keras)
  include_directories(${TENSORFLOW_LIB_DIR}/include)
  add_definitions(-DENABLE_KERAS=1)
endif(ENABLE_KERAS)

# build python lib
if(BUILD_PYTHON_LIB)
  # pybind11
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/third_party/pybind11)
  # if(USE_FOLDERS) set_target_properties (pybind11 PROPERTIES FOLDER
  # third_party) endif()
  
  # pyinfer
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/py_fwd)
endif()

# fwd test
if(ENABLE_INFER_TESTS)
  # OpenCV
  list(APPEND CMAKE_PREFIX_PATH ${OPENCV_CMAKE_PATH})
  find_package(OpenCV)
  if(NOT OpenCV_FOUND)
    message(
      FATAL_ERROR
        "Please set OpenCV cmake path to OPENCV_CMAKE_PATH,
        like -DOPENCV_CMAKE_PATH=D:/opencv/build
        or -DOPENCV_CMAKE_PATH=/usr/local/share/OpenCV")
  endif()
  include_directories(${OpenCV_INCLUDE_DIRS})

  # infer test
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/test_fwd)
endif(ENABLE_INFER_TESTS)

# unit tests
if(ENABLE_UNIT_TESTS)
  # gtest
  include_directories(
    ${PROJECT_SOURCE_DIR}/source/third_party/googletest/include)
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/third_party/googletest)
  set_target_properties(gtest PROPERTIES FOLDER third_party)
  set_target_properties(gtest_main PROPERTIES FOLDER third_party)

  # unit_test
  add_subdirectory(${PROJECT_SOURCE_DIR}/source/unit_test)
endif()
