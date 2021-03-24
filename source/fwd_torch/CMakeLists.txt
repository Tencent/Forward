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

file(GLOB TORCH_LIB_SOURCES torch_engine/*.h torch_engine/*.cpp)
file(GLOB TORCH_CVT_SOURCES torch_cvt/*.h torch_cvt/*.cpp)
file(GLOB TORCH_CRT_SOURCES torch_cvt/torch_desc_creators/*.h)
file(GLOB TORCH_PASS_SOURCES torch_cvt/torch_passes/*.h torch_cvt/torch_passes/*.cpp)

add_library(fwd_torch 
                ${TORCH_LIB_SOURCES}
                ${TORCH_CVT_SOURCES}
                ${TORCH_CRT_SOURCES}
                ${TORCH_PASS_SOURCES})

source_group("engine" FILES ${TORCH_LIB_SOURCES})
source_group("converter" FILES ${TORCH_CVT_SOURCES})
source_group("creators" FILES ${TORCH_CRT_SOURCES})
source_group("passes" FILES ${TORCH_PASS_SOURCES})

if (ENABLE_TENSORRT)
    target_link_libraries(fwd_torch PUBLIC trt_engine)
endif(ENABLE_TENSORRT)

set_target_properties(fwd_torch PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

set_property(TARGET fwd_torch PROPERTY CXX_STANDARD 14)
