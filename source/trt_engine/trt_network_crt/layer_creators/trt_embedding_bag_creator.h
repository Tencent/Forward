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
#include "trt_engine/trt_network_crt/plugins/embedding_bag_plugin/embedding_bag_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Embedding Bag创建器
 */
template <>
class TLayerCreator<TrtEmbeddingBagDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtEmbeddingBagDesc::CreateLayer";
    const auto embedding_bag_desc = dynamic_cast<const TrtEmbeddingBagDesc*>(layer_desc);
    T_CHECK(embedding_bag_desc);

    // input + offset or input only
    T_CHECK_LE(input_tensors.size(), 2);

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        EMBEDDING_BAG_PLUGIN_NAME, EMBEDDING_BAG_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("dim", &embedding_bag_desc->dim, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("count", &embedding_bag_desc->count, nvinfer1::PluginFieldType::kINT32,
                            1);
    field_data.emplace_back("offset", &embedding_bag_desc->offset,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("op", &embedding_bag_desc->op, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("data", embedding_bag_desc->data.Data(),
                            nvinfer1::PluginFieldType::kFLOAT32,
                            embedding_bag_desc->dim * embedding_bag_desc->count);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("embedding_bag", &plugin_data));

    // add the plugin to the TensorRT network
    const auto embedding_bag =
        network->addPluginV2(input_tensors.data(), input_tensors.size(), *plugin_obj);

    if (embedding_bag == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [embedding_bag] layer.";
      return {};
    }

    embedding_bag->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Embedding Bag]")).c_str());
    return {embedding_bag->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
