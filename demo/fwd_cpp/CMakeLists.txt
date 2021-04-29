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
# Authors: Aster JIAN (asterjian@qq.com)
#          Yzx (yzxyzxyzx777@outlook.com)
#          Ao LI (346950981@qq.com)
#          Paul LU (lujq96@gmail.com)
cmake_minimum_required(VERSION 3.10.1 FATAL_ERROR)

project(Forward LANGUAGES C CXX CUDA)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Enable Torch
option(ENABLE_TORCH "Enable Torch" OFF)
# Enable TensorFlow
option(ENABLE_TENSORFLOW "Enable TensorFlow" OFF)

# CUDA
find_package(CUDA)

# TensorRT
find_package(TensorRT)

if(NOT TensorRT_FOUND) # use sources
  message(
    FATAL_ERROR
      "Can NOT find TensorRT, you may set TensorRT root via TensorRT_ROOT.
            like -DTensorRT_ROOT=D:/Libs/TensorRT-6.0.1.5/")
endif()

include_directories(${TensorRT_INCLUDE_DIRS})
# include headers
include_directories(${PROJECT_SOURCE_DIR}/include)
find_library(FWD_ENGINE trt_engine HINTS ${PROJECT_SOURCE_DIR}/libs)

CUDA_ADD_EXECUTABLE(test_fwd_engine test_fwd_engine.cpp)
target_link_libraries(test_fwd_engine
        ${TensorRT_LIBRARIES}
        ${FWD_ENGINE}
        )
CUDA_ADD_CUBLAS_TO_TARGET(test_fwd_engine)
set_property(TARGET test_fwd_engine PROPERTY CXX_STANDARD 14)

if (ENABLE_TORCH)
# libtorch
find_package(Torch)
if(NOT TORCH_FOUND)
  message(
    FATAL_ERROR
      "Please set libtorch/share/cmake/ path to TORCH_CMAKE_PATH,
          like -DTORCH_CMAKE_PATH=D:/libtorch/share/cmake/Torch
          or -DTORCH_CMAKE_PATH=/usr/local/share/cmake")
endif()

CUDA_ADD_EXECUTABLE(test_fwd_torch test_fwd_torch.cpp)
find_library(FWD_TORCH fwd_torch HINTS ${PROJECT_SOURCE_DIR}/libs)
target_link_libraries(test_fwd_torch
        ${FWD_TORCH}
        ${TensorRT_LIBRARIES}
        ${TORCH_LIBRARIES}
        )
CUDA_ADD_CUBLAS_TO_TARGET(test_fwd_torch)
set_property(TARGET test_fwd_torch PROPERTY CXX_STANDARD 14)
endif(ENABLE_TORCH)

if (ENABLE_TENSORFLOW)
if (NOT Tensorflow_ROOT)
  set(Tensorflow_ROOT ${PROJECT_SOURCE_DIR}/tensorflow) 
endif()
message(STATUS "Tensorflow_ROOT = ${Tensorflow_ROOT}")
#tensorflow
include_directories(${Tensorflow_ROOT}/include)
find_library(TF_LIBRARY tensorflow HINTS ${Tensorflow_ROOT}/lib)

CUDA_ADD_EXECUTABLE(test_fwd_tf test_fwd_tf.cpp)
find_library(FWD_TF fwd_tf HINTS ${PROJECT_SOURCE_DIR}/libs)
target_link_libraries(test_fwd_tf
        ${TF_LIBRARY}
        ${FWD_TF}
        ${TensorRT_LIBRARIES}
        )
CUDA_ADD_CUBLAS_TO_TARGET(test_fwd_tf)
set_property(TARGET test_fwd_tf PROPERTY CXX_STANDARD 14)
endif(ENABLE_TENSORFLOW)


