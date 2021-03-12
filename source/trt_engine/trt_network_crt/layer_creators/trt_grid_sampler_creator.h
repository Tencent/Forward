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
#include "trt_engine/trt_network_crt/plugins/grid_sampler_plugin/grid_sampler_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Grid Sampler层创建器
 */
template <>
class TLayerCreator<TrtGridSamplerDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtGridSamplerDesc::CreateLayer";

    const auto grid_sampler_desc = dynamic_cast<const TrtGridSamplerDesc*>(layer_desc);
    T_CHECK(grid_sampler_desc);

    T_CHECK_EQ(input_tensors.size(), 2);
    const auto input = input_tensors[0];
    const auto grid = input_tensors[1];

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        GRID_SAMPLER_PLUGIN_NAME, GRID_SAMPLER_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("interpolation_mode", &grid_sampler_desc->interpolation_mode,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("padding_mode", &grid_sampler_desc->padding_mode,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("align_corners", &grid_sampler_desc->align_corners,
                            nvinfer1::PluginFieldType::kINT32, 1);

    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("grid_sampler", &plugin_data));

    // add the plugin to the TensorRT network
    ITensorVector tensor_inputs{input, grid};
    const auto grid_sampler =
        network->addPluginV2(tensor_inputs.data(), tensor_inputs.size(), *plugin_obj);

    if (grid_sampler == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [grid_sampler] layer.";
      return {};
    }

    grid_sampler->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Grid Sampler]")).c_str());
    return {grid_sampler->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
