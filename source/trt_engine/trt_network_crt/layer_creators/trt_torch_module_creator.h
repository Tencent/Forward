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

#include <vector>

#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"
#include "trt_engine/trt_network_crt/plugins/torch_module_plugin/torch_module_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT TorchModule Plugin层创建器
 */
template <>
class TLayerCreator<TrtTorchModuleDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtTorchModuleDesc::CreateLayer";
    const auto torch_module_desc = dynamic_cast<const TrtTorchModuleDesc*>(layer_desc);
    T_CHECK(torch_module_desc);

    // Check if the original module file TORCH_MODULE_PLUGIN_MODULE_PATH exist. If it is not exist,
    // then copy the original module from the original module path.
    // The torch module plugin will automatically load the original module from
    // TORCH_MODULE_PLUGIN_MODULE_PATH as the referred module to build a new sub_module.
    if (!TrtCommon::CheckAndCopyFile(TORCH_MODULE_PLUGIN_MODULE_PATH,
                                     torch_module_desc->module_path))
      return {};

    const auto& node_ids = torch_module_desc->node_ids;
    const auto& in_types = torch_module_desc->in_types;
    const auto& out_types = torch_module_desc->out_types;
    const auto& out_dims = torch_module_desc->out_dims;

    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        TORCH_MODULE_PLUGIN_NAME, TORCH_MODULE_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("data_type", &dtype, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("node_ids", node_ids.data(), nvinfer1::PluginFieldType::kINT32,
                            node_ids.size());
    field_data.emplace_back("in_types", in_types.data(), nvinfer1::PluginFieldType::kINT32,
                            in_types.size());
    field_data.emplace_back("out_types", out_types.data(), nvinfer1::PluginFieldType::kINT32,
                            out_types.size());
    field_data.emplace_back("out_dims", out_dims.data(), nvinfer1::PluginFieldType::kDIMS,
                            out_dims.size());
    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("torch_module", &plugin_data));

    // add the plugin to the TensorRT network
    const auto torch_plugin =
        network->addPluginV2(input_tensors.data(), input_tensors.size(), *plugin_obj);

    if (torch_plugin == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [TorchModulePlugin] layer.";
      return {};
    }

    torch_plugin->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [TorchModulePlugin]")).c_str());

    ITensorVector outputs;
    for (int i = 0; i < torch_plugin->getNbOutputs(); ++i) {
      outputs.push_back(torch_plugin->getOutput(i));
    }
    return outputs;
  }
};

FWD_TRT_NAMESPACE_END
