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

CUDA_ADD_EXECUTABLE(test_fwd test_fwd.cpp image.h image.cpp)

target_link_libraries(test_fwd
        fwd_torch
        ${TensorRT_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS})

CUDA_ADD_CUBLAS_TO_TARGET(test_fwd)

set_property(TARGET test_fwd PROPERTY CXX_STANDARD 14)
