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

// NOTE: TensorRT 7.0 及其以下版本对 Resize 支持的不是那么完善
// 这个 plugin 针对 7.0 及其以下版本的 TensorRT 实现 Upsample Bilinear 2D

#include <string>
#include <vector>

#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"
#include "trt_engine/trt_network_crt/plugins/upsampler_plugin/upsample_bilinear_2d.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Upsample Bilinear 2D层创建器
 */
template <>
class TLayerCreator<TrtUpsampleBilinearDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtUpsampleBilinearDesc::CreateLayer";
    const auto upsample_bilinear_desc = dynamic_cast<const TrtUpsampleBilinearDesc*>(layer_desc);
    T_CHECK(upsample_bilinear_desc);

    auto input = input_tensors[0];

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        UPSAMPLE_BILINEAR_2D_PLUGIN_NAME, UPSAMPLE_BILINEAR_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("output_h", &upsample_bilinear_desc->output_h,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("output_w", &upsample_bilinear_desc->output_w,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("align_corners", &upsample_bilinear_desc->alignCorners,
                            nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("scale_h", &upsample_bilinear_desc->scale_h,
                            nvinfer1::PluginFieldType::kFLOAT32, 1);
    field_data.emplace_back("scale_w", &upsample_bilinear_desc->scale_w,
                            nvinfer1::PluginFieldType::kFLOAT32, 1);
    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("upsample_bilinear", &plugin_data));

    // add the plugin to the TensorRT network
    const auto upsample_bilinear = network->addPluginV2(&input, 1, *plugin_obj);

    if (upsample_bilinear == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [upsample_bilinear] layer.";
      return {};
    }

    upsample_bilinear->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Upsample Bilinear]")).c_str());
    return {upsample_bilinear->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
