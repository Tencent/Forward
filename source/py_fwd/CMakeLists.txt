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

file(GLOB BINDING_HEADERS *.h)

pybind11_add_module(forward py_forward.cpp)

# target_link_libraries(forward PUBLIC -static-libstdc++ -static-libgcc)
set_target_properties(forward PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
# pybind11::overload_cast<...> requires compiling in C++14 mode
set_property(TARGET forward PROPERTY CXX_STANDARD 14)

if (ENABLE_TORCH)
    target_link_libraries(forward PUBLIC fwd_torch ${TORCH_LIBRARIES} ${TorchPath}/lib/libtorch_python.so)
endif(ENABLE_TORCH)

if (ENABLE_TENSORFLOW)
    message(STATUS "py_forward link to PyTensorflow libs : ${PY_TENSORFLOW_LIB}")
    target_link_libraries(forward PUBLIC fwd_tf ${PY_TENSORFLOW_LIB})
endif(ENABLE_TENSORFLOW)

if (ENABLE_KERAS)
    target_link_libraries(forward PUBLIC fwd_keras hdf5_cpp nlohmann_json::nlohmann_json)
endif(ENABLE_KERAS)
