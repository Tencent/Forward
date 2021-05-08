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

if (ENABLE_RNN)
    message("TensorRT Version : ${TensorRT_VERSION_MAJOR}")
    if (${TensorRT_VERSION_MAJOR} LESS "7")
        add_definitions(-DSUPPORT_RNN=0)
    endif()
endif(ENABLE_RNN)


set(PLUGIN_HEADERS)
set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    adaptive_lin_plugin
    adaptive_pooling_plugin
    qkv_to_context_plugin  
    cast_plugin
    constant_pad_plugin
    common
    embedding_bag_plugin
    emb_layer_norm_plugin
    gelu_plugin
    grid_sampler_plugin
    index_plugin
    norm_plugin
    normalization_plugin
    reduce_plugin
    reflection_padding_plugin
    skip_layer_norm_plugin
    split_plugin
    upsampler_plugin
    )

file(GLOB TRT_ENGINE_SOURCES trt_engine/*.h trt_engine/*.cpp)
file(GLOB TRT_COMMON_SOURCES trt_common/*.h trt_common/*.cpp)
file(GLOB TRT_NET_CRT_SOURCES trt_network_crt/*.h trt_network_crt/*.cpp)
file(GLOB TRT_NET_CRT_HEADERS trt_network_crt/layer_creators/*.h)

set(PLUGIN_DIR "trt_network_crt/plugins/")

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_DIR}${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_DIR}${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

cuda_add_library(trt_engine
            ${TRT_ENGINE_SOURCES}
            ${TRT_COMMON_SOURCES}
            ${TRT_NET_CRT_SOURCES}
            ${TRT_NET_CRT_HEADERS}
            ${PLUGIN_HEADERS}
            ${PLUGIN_SOURCES}
            ${PLUGIN_CU_SOURCES})

source_group("engine"       FILES ${TRT_ENGINE_SOURCES})
source_group("common"       FILES ${TRT_COMMON_SOURCES})
source_group("network_crt"  FILES ${TRT_NET_CRT_SOURCES})
source_group("network_crt/creators"  FILES ${TRT_NET_CRT_HEADERS})
source_group("network_crt/plugins/headers"  FILES ${PLUGIN_HEADERS})
source_group("network_crt/plugins/cpp"  FILES ${PLUGIN_SOURCES})
source_group("network_crt/plugins/cu"  FILES ${PLUGIN_CU_SOURCES})

target_link_libraries(trt_engine ${TensorRT_LIBRARIES} simple-utils)

if (MSVC)
    target_link_libraries(trt_engine cudnn)
	CUDA_ADD_CUBLAS_TO_TARGET(trt_engine)
endif (MSVC)

set_target_properties(trt_engine PROPERTIES CUDA_ARCHITECTURES "${GPU_ARCHS}")

set_target_properties(trt_engine PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

set_property(TARGET trt_engine PROPERTY CXX_STANDARD 14)
