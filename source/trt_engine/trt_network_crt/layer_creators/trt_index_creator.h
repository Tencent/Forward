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
#include "trt_engine/trt_network_crt/plugins/index_plugin/index_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Index创建器
 */
template <>
class TLayerCreator<TrtIndexDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtIndexDesc::CreateLayer";

    const auto index_desc = dynamic_cast<const TrtIndexDesc*>(layer_desc);
    T_CHECK(index_desc);

    T_CHECK_EQ(input_tensors.size(), 1);
    const auto input = input_tensors[0];

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator =
        getPluginRegistry()->getPluginCreator(INDEX_PLUGIN_NAME, INDEX_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("nbDims", &index_desc->nbDims, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("nbIndexDims", &index_desc->nbIndexDims,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("nbIndex", &index_desc->nbIndex, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("data", index_desc->indices.data(), nvinfer1::PluginFieldType::kINT32,
                            index_desc->nbIndex * index_desc->nbIndexDims);
    field_data.emplace_back("pos", index_desc->axis.data(), nvinfer1::PluginFieldType::kINT32,
                            index_desc->nbDims);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("index", &plugin_data));

    // add the plugin to the TensorRT network
    ITensorVector tensor_inputs{input};
    const auto index =
        network->addPluginV2(tensor_inputs.data(), tensor_inputs.size(), *plugin_obj);

    if (index == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [embedding_bag] layer.";
      return {};
    }

    index->setName((std::to_string(network->getNbLayers()) + std::string(" [Index]")).c_str());
    return {index->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
