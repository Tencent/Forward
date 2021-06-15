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

file(GLOB KERAS_LIB_SOURCES keras_engine/*.h keras_engine/*.cpp)
file(GLOB KERAS_CVT_SOURCES keras_cvt/*.h keras_cvt/*.cpp)
file(GLOB KERAS_CRT_SOURCES keras_cvt/keras_desc_creators/*.h)

add_library(fwd_keras 
                ${KERAS_LIB_SOURCES}
                ${KERAS_CVT_SOURCES}
                ${KERAS_CRT_SOURCES})

source_group("engine"       FILES ${KERAS_LIB_SOURCES})
source_group("converter"    FILES ${KERAS_CVT_SOURCES})
source_group("creators"     FILES ${KERAS_CRT_SOURCES})

target_link_libraries(fwd_keras PUBLIC hdf5_cpp)

if (ENABLE_TENSORRT)
    target_link_libraries(fwd_keras PUBLIC trt_engine)
endif(ENABLE_TENSORRT)

set_target_properties(fwd_keras PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

set_property(TARGET fwd_keras PROPERTY CXX_STANDARD 14)

