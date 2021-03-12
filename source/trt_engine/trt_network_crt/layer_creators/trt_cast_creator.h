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
#include "trt_engine/trt_network_crt/plugins/cast_plugin/cast_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Cast 层创建器
 */
template <>
class TLayerCreator<TrtCastDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtCastDesc::CreateLayer";
    const TrtCastDesc* const cast_desc = dynamic_cast<const TrtCastDesc*>(layer_desc);
    T_CHECK(cast_desc);

    nvinfer1::ITensor* input = input_tensors[0];

    nvinfer1::DataType input_type = network->getInput(0)->getType();
    if (input_type == cast_desc->otype) return {input};

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator =
        getPluginRegistry()->getPluginCreator(CAST_PLUGIN_NAME, CAST_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("output_type", &cast_desc->otype, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj =
        TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("cast", &plugin_data));

    // add the plugin to the TensorRT network
    nvinfer1::IPluginV2Layer* cast = network->addPluginV2(&input, 1, *plugin_obj);

    if (cast == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [cast] layer.";
      return {};
    }

    cast->setName((std::to_string(network->getNbLayers()) + std::string(" [Cast]")).c_str());
    return {cast->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
