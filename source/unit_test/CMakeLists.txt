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

file(GLOB TEST_HEADER_FILES *.h *.hpp)

cuda_add_executable(unit_test unit_test.cpp ${TEST_HEADER_FILES})

target_link_libraries(unit_test 
        gtest
        ${TensorRT_LIBRARIES}
)
CUDA_ADD_CUBLAS_TO_TARGET(unit_test)

if (ENABLE_TENSORFLOW)
    target_link_libraries(unit_test fwd_tf tensorflow)
endif(ENABLE_TENSORFLOW)

if (ENABLE_TORCH)
    target_link_libraries(unit_test fwd_torch ${TORCH_LIBRARIES})
endif(ENABLE_TORCH)

if (ENABLE_KERAS)
    target_link_libraries(unit_test fwd_keras tensorflow)
endif(ENABLE_KERAS)
