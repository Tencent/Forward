/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <NvInferPlugin.h>

#include <string>
#include <vector>

#include "trt_engine/trt_network_crt/plugins/common/plugin_common.h"

FWD_TRT_NAMESPACE_BEGIN

static constexpr int MAX_GELU_VAL{10};

nvinfer1::ITensor* CreateGeluLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                                   bool use_fp16, bool use_int8);

nvinfer1::ITensor* CreateGeluCombinattion(nvinfer1::INetworkDefinition* network,
                                          nvinfer1::ITensor* input);

nvinfer1::ITensor* CreateGeluPlugin(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                                    bool use_fp16);

FWD_TRT_NAMESPACE_END
