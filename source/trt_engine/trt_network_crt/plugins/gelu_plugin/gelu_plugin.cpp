// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.
//
// ¨X¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨[
// ¨U©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨€¨€¨[©¤©¤©¤©¤©¤©¤¨€¨€¨[©¤©¤©¤¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨U
// ¨U©¤©¤¨€¨€¨X¨T¨T¨T¨T¨T¨T¨a©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨€¨€¨U©¤©¤©¤©¤©¤©¤¨€¨€¨U©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨U
// ¨U©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨[©¤©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨X¨a©¤©¤¨€¨€¨U©¤©¤¨€¨[©¤©¤¨€¨€¨U©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨€¨U©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨X¨a©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨U
// ¨U©¤©¤¨€¨€¨X¨T¨T¨T¨T¨T¨a©¤©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨€¨€¨U¨€¨€¨€¨€¨€¨[¨€¨€¨U©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨U©¤©¤¨€¨€¨X¨T¨T¨T¨T¨€¨€¨[©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨U
// ¨U©¤©¤¨€¨€¨U©¤©¤©¤©¤©¤©¤©¤©¤©¤¨^¨€¨€¨€¨€¨€¨€¨€¨X¨a©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨^¨€¨€¨€¨€¨X¨€¨€¨€¨€¨X¨a©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨€¨€¨U©¤©¤©¤©¤¨€¨€¨U©¤©¤¨€¨€¨€¨€¨€¨€¨€¨€¨X¨a©¤©¤¨U
// ¨U©¤©¤¨^¨T¨a©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤¨^¨T¨T¨T¨T¨T¨T¨a©¤©¤©¤¨^¨T¨a©¤©¤©¤©¤¨^¨T¨a©¤©¤©¤¨^¨T¨T¨T¨a¨^¨T¨T¨T¨a©¤©¤©¤¨^¨T¨a©¤©¤©¤©¤¨^¨T¨a©¤©¤¨^¨T¨a©¤©¤©¤©¤¨^¨T¨a©¤©¤¨^¨T¨T¨T¨T¨T¨T¨T¨a©¤©¤©¤¨U
// ¨^¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨a
//
// Authors: Aster JIAN (asterjian@qq.com)
//          Yzx (yzxyzxyzx777@outlook.com)
//          Ao LI (346950981@qq.com)
//          Paul LU (lujq96@gmail.com)
#include "trt_engine/trt_network_crt/plugins/gelu_plugin/gelu_plugin.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/plugins/common/serialize.hpp"

FWD_TRT_NAMESPACE_BEGIN

constexpr float f3_{3.0f};
constexpr float f004_{0.044715f};
constexpr float f079_{0.79788456080286535587989211986876f};
constexpr float f1_{1.0f};
constexpr float f05_{0.5f};

nvinfer1::ITensor* CreateGeluLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                                   bool use_fp16, bool use_int8) {
  if (use_int8) {
    return CreateGeluCombinattion(network, input);
  }
  return CreateGeluPlugin(network, input, use_fp16);
}

nvinfer1::ITensor* CreateGeluCombinattion(nvinfer1::INetworkDefinition* network,
                                          nvinfer1::ITensor* input) {
  const auto nbdims = input->getDimensions().nbDims;
  const nvinfer1::Dims ref_dims = {nbdims, 1, 1, 1, 1, 1, 1, 1, 1};
  const nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;

  auto POW = network->addConstant(ref_dims, {dtype, &f3_, 1})->getOutput(0);
  auto MULTIPLY = network->addConstant(ref_dims, {dtype, &f004_, 1})->getOutput(0);
  auto SQRT = network->addConstant(ref_dims, {dtype, &f079_, 1})->getOutput(0);
  auto ONE = network->addConstant(ref_dims, {dtype, &f1_, 1})->getOutput(0);
  auto HALF = network->addConstant(ref_dims, {dtype, &f05_, 1})->getOutput(0);

  using EleWiseOp = nvinfer1::ElementWiseOperation;

  auto X_pow = network->addElementWise(*input, *POW, EleWiseOp::kPOW);
  auto X_mul = network->addElementWise(*X_pow->getOutput(0), *MULTIPLY, EleWiseOp::kPROD);
  auto X_add = network->addElementWise(*input, *X_mul->getOutput(0), EleWiseOp::kSUM);
  auto X_sqrt = network->addElementWise(*X_add->getOutput(0), *SQRT, EleWiseOp::kPROD);
  auto X_tanh = network->addActivation(*X_sqrt->getOutput(0), nvinfer1::ActivationType::kTANH);
  auto X_one = network->addElementWise(*X_tanh->getOutput(0), *ONE, EleWiseOp::kSUM);
  auto CDF = network->addElementWise(*X_one->getOutput(0), *HALF, EleWiseOp::kPROD);
  auto gelu_layer = network->addElementWise(*CDF->getOutput(0), *input, EleWiseOp::kPROD);
  fwd::TrtCommon::SetOutputRange(gelu_layer, MAX_GELU_VAL);
  return gelu_layer->getOutput(0);
}

nvinfer1::ITensor* CreateGeluPlugin(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                                    bool use_fp16) {
  auto dtype = use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;

  nvinfer1::IPluginCreator* creator =
      getPluginRegistry()->getPluginCreator("CustomGeluPluginDynamic", "1");
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("type_id", &dtype, nvinfer1::PluginFieldType::kINT32, 1);

  const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                    field_data.data()};
  const auto plugin_obj = fwd::TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
      creator->createPlugin("gelu", &plugin_data));

  nvinfer1::IPluginV2Layer* gelu_layer = network->addPluginV2(&input, 1, *plugin_obj);
  T_CHECK(gelu_layer);

  fwd::TrtCommon::SetOutputRange(gelu_layer, fwd::trt_::MAX_GELU_VAL);
  return gelu_layer->getOutput(0);
}

FWD_TRT_NAMESPACE_END
