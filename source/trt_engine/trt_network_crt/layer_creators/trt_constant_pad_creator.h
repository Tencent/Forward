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

#include "common/trt_utils.h"
#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"
#include "trt_engine/trt_network_crt/plugins/constant_pad_plugin/constant_pad_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Padding层创建器
 */
template <>
class TLayerCreator<TrtConstantPadDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtConstantPadDesc::CreateLayer";
    const auto constant_pad_desc = dynamic_cast<const TrtConstantPadDesc*>(layer_desc);
    T_CHECK(constant_pad_desc);

    auto input = input_tensors[0];
    // 检查输入维度
    auto dims = constant_pad_desc->dims;
    if (dims.size() == 4 && input->getDimensions().nbDims == 5) {
      // 填充到 3D
      dims.push_back(0);
      dims.push_back(0);
    }
    if (constant_pad_desc->value == 0.0f && dims.size() == 4) {
      LOG(INFO) << "Zero 2D padding";
      const auto zero_pad = network->addPadding(*input, TrtUtils::ToDimsHW({dims[2], dims[0]}),
                                                TrtUtils::ToDimsHW({dims[3], dims[1]}));
      return {zero_pad->getOutput(0)};
    }
    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        CONSTANT_PAD_PLUGIN_NAME, CONSTANT_PAD_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("padding_dims", dims.data(), nvinfer1::PluginFieldType::kINT32,
                            dims.size());
    field_data.emplace_back("constant", &constant_pad_desc->value,
                            nvinfer1::PluginFieldType::kFLOAT32, 1);
    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);
    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("constant_pad", &plugin_data));
    // add the plugin to the TensorRT network
    const auto constant_pad = network->addPluginV2(&input, 1, *plugin_obj);

    if (constant_pad == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [constant_pad] layer.";
      return {};
    }

    constant_pad->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Constant Pad]")).c_str());
    return {constant_pad->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
