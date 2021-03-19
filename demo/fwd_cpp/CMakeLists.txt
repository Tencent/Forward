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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/cmake")

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

# libtorch
find_package(Torch)
if(NOT TORCH_FOUND)
  message(
    FATAL_ERROR
      "Please set libtorch/share/cmake/ path to TORCH_CMAKE_PATH,
          like -DTORCH_CMAKE_PATH=D:/libtorch/share/cmake/Torch
          or -DTORCH_CMAKE_PATH=/usr/local/share/cmake")
endif()

CUDA_ADD_EXECUTABLE(test_fwd test_fwd.cpp)

find_library(FWD_LIBRARIES NAMES fwd_torch PATHS "${PROJECT_SOURCE_DIR}/libs")
message("FWD_LIBRARIES = ${FWD_LIBRARIES}")

target_link_libraries(test_fwd
        ${FWD_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${TORCH_LIBRARIES}
        )

CUDA_ADD_CUBLAS_TO_TARGET(test_fwd)

set_property(TARGET test_fwd PROPERTY CXX_STANDARD 14)
