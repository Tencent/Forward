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
// ╔════════════════════════════════════════════════════════════════════════════════════════╗
// ║──█████████╗───███████╗───████████╗───██╗──────██╗───███████╗───████████╗───████████╗───║
// ║──██╔══════╝──██╔════██╗──██╔════██╗──██║──────██║──██╔════██╗──██╔════██╗──██╔════██╗──║
// ║──████████╗───██║────██║──████████╔╝──██║──█╗──██║──█████████║──████████╔╝──██║────██║──║
// ║──██╔═════╝───██║────██║──██╔════██╗──██║█████╗██║──██╔════██║──██╔════██╗──██║────██║──║
// ║──██║─────────╚███████╔╝──██║────██║──╚████╔████╔╝──██║────██║──██║────██║──████████╔╝──║
// ║──╚═╝──────────╚══════╝───╚═╝────╚═╝───╚═══╝╚═══╝───╚═╝────╚═╝──╚═╝────╚═╝──╚═══════╝───║
// ╚════════════════════════════════════════════════════════════════════════════════════════╝
//
// Authors: Aster JIAN (asterjian@qq.com)
//          Yzx (yzxyzxyzx777@outlook.com)
//          Ao LI (346950981@qq.com)
//          Paul LU (lujq96@gmail.com)

#pragma once

#include <string>
#include <vector>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"
#include "trt_engine/trt_network_crt/plugins/adaptive_pooling_plugin/adaptive_pooling_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Adaptive Pooling 层创建器
 */
template <>
class TLayerCreator<TrtAdaptivePoolDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtAdaptivePoolDesc::CreateLayer";
    const auto adaptive_pool_desc = dynamic_cast<const TrtAdaptivePoolDesc*>(layer_desc);
    T_CHECK(adaptive_pool_desc);
    auto input = input_tensors[0];

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        ADAPTIVE_POOLING_PLUGIN_NAME, ADAPTIVE_POOLING_PLUGIN_VERSION);
    int pool_type = adaptive_pool_desc->pooling_type == nvinfer1::PoolingType::kMAX ? 0 : 1;
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("output_size", adaptive_pool_desc->output_size.data(),
                            nvinfer1::PluginFieldType::kINT32,
                            adaptive_pool_desc->output_size.size());
    field_data.emplace_back("type", &pool_type, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("adaptive_pool", &plugin_data));

    // add the plugin to the TensorRT network
    const auto adaptive_pool = network->addPluginV2(&input, 1, *plugin_obj);

    if (adaptive_pool == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [adaptive_pool] layer.";
      return {};
    }

    adaptive_pool->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Adaptive Pool]")).c_str());
    return {adaptive_pool->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
