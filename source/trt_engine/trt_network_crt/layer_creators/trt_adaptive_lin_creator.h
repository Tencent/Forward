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
#include "trt_engine/trt_network_crt/plugins/adaptive_lin_plugin/adaptive_lin_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Normalization层创建器
 * Normalization层不会传入running_mean/variance信息，通过内部层计算较慢
 * 故一般使用plugin进行操作
 */
template <>
class TLayerCreator<TrtAdaptiveLinDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtNormalizationDesc::CreateLayer";

    const auto norm_desc = dynamic_cast<const TrtAdaptiveLinDesc*>(layer_desc);
    T_CHECK(norm_desc);

    LOG(INFO) << "TrtNormalizationDesc::CreateAdaptiveLIN";

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        ADAPTIVE_LIN_PLUGIN_NAME, ADAPTIVE_LIN_PLUGIN_VERSION);

    // input0 : input
    // input1 : layer norm rho
    // input2 : instance norm rho
    const auto input_dim = input_tensors[0]->getDimensions();
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("input_dim", &input_dim, nvinfer1::PluginFieldType::kDIMS, 1);
    field_data.emplace_back("epsilon", &norm_desc->epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("max_batch_size", &norm_desc->max_batch_size,
                            nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("adaptive_lin", &plugin_data));

    // add the plugin to the TensorRT network
    const auto lin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);

    if (lin == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [adaptive_lin] layer.";
      return {};
    }

    lin->setName((std::to_string(network->getNbLayers()) + std::string(" [Adaptive LIN]")).c_str());

    // merge
    const auto ln = network->addElementWise(*lin->getOutput(0), *input_tensors[1],
                                            nvinfer1::ElementWiseOperation::kPROD);
    const auto in = network->addElementWise(*lin->getOutput(1), *input_tensors[2],
                                            nvinfer1::ElementWiseOperation::kPROD);
    const auto element = network->addElementWise(*ln->getOutput(0), *in->getOutput(0),
                                                 nvinfer1::ElementWiseOperation::kSUM);
    return {element->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
