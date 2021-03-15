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
#include "trt_engine/trt_network_crt/plugins/normalization_plugin/normalization_plugin.h"
#include "trt_engine/trt_network_crt/plugins/skip_layer_norm_plugin/skip_layer_norm_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Normalization层创建器
 * Normalization层不会传入running_mean/variance信息，通过内部层计算较慢
 * 故一般使用plugin进行操作
 */
template <>
class TLayerCreator<TrtNormalizationDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtNormalizationDesc::CreateLayer";

    const auto norm_desc = dynamic_cast<const TrtNormalizationDesc*>(layer_desc);
    T_CHECK(norm_desc);

    switch (norm_desc->type) {
      case TrtNormalizationType::SKIP_LAYER_NORMALIZATION:
        return CreateSkipLayerNorm(network, layer_desc, input_tensors);
      default:
        return CreatePluginImpl(network, layer_desc, input_tensors);
    }
  }

 private:
  ITensorVector CreatePluginImpl(nvinfer1::INetworkDefinition* network,
                                 const TrtLayerDesc* layer_desc,
                                 const ITensorVector& input_tensors) const {
    LOG(INFO) << "TrtNormalizationDesc::CreatePluginImpl";

    const auto norm_desc = dynamic_cast<const TrtNormalizationDesc*>(layer_desc);
    T_CHECK(norm_desc);

    auto input = input_tensors[0];

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        NORMALIZATION_PLUGIN_NAME, NORMALIZATION_PLUGIN_VERSION);

    std::vector<nvinfer1::PluginField> field_data;
    const nvinfer1::PluginFieldType field_type = nvinfer1::PluginFieldType::kFLOAT32;
    std::vector<float> ones, zeros;
    if (!norm_desc->affine) {
      const int n_channels = input->getDimensions().d[1];
      ones.resize(n_channels, 1.0f);
      zeros.resize(n_channels, 0.0f);
      field_data.emplace_back("epsilon", &norm_desc->epsilon, field_type, 1);
      field_data.emplace_back("scales", ones.data(), field_type, ones.size());
      field_data.emplace_back("bias", zeros.data(), field_type, zeros.size());
    } else {
      field_data.emplace_back("epsilon", &norm_desc->epsilon, field_type, 1);
      field_data.emplace_back("scales", norm_desc->scales.Data(), field_type,
                              norm_desc->scales.Count());
      field_data.emplace_back("bias", norm_desc->bias.Data(), field_type, norm_desc->bias.Count());
    }
    if (norm_desc->use_input_stats) {
      field_data.emplace_back("running_mean", nullptr, field_type, 0);
      field_data.emplace_back("running_var", nullptr, field_type, 0);
    } else {
      field_data.emplace_back("running_mean", norm_desc->running_mean.Data(), field_type,
                              norm_desc->running_mean.Count());
      field_data.emplace_back("running_var", norm_desc->running_var.Data(), field_type,
                              norm_desc->running_var.Count());
    }

    field_data.emplace_back("type", &norm_desc->type, nvinfer1::PluginFieldType::kINT32, 1);

    auto input_type =
        TrtCommon::GetDataType(norm_desc->use_fp16, norm_desc->use_int8, norm_desc->use_calib_mode);
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("max_batch_size", &norm_desc->max_batch_size,
                            nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("normalization", &plugin_data));

    // add the plugin to the TensorRT network
    const auto instance_norm = network->addPluginV2(&input, 1, *plugin_obj);

    if (instance_norm == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [normalization] layer.";
      return {};
    }

    instance_norm->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Normalization]")).c_str());
    return {instance_norm->getOutput(0)};
  }

  ITensorVector CreateSkipLayerNorm(nvinfer1::INetworkDefinition* network,
                                    const TrtLayerDesc* layer_desc,
                                    const ITensorVector& input_tensors) const {
    LOG(INFO) << "TrtNormalizationDesc::CreateSkipLayerNorm";

    const TrtNormalizationDesc* const norm_desc =
        dynamic_cast<const TrtNormalizationDesc*>(layer_desc);

    // 创建 Plugin
    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        bert::SKIP_LAYER_NORM_NAME, bert::SKIP_LAYER_NORM_VERSION);

    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("ld", &norm_desc->leading_dim, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("beta", norm_desc->bias.Data(), nvinfer1::PluginFieldType::kFLOAT32,
                            norm_desc->bias.Count());
    field_data.emplace_back("gamma", norm_desc->scales.Data(), nvinfer1::PluginFieldType::kFLOAT32,
                            norm_desc->scales.Count());

    const auto dtype = TrtCommon::GetDataType(norm_desc->use_fp16, norm_desc->use_int8, true);
    field_data.emplace_back("type_id", &dtype, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("skip_layer_norm", &plugin_data));

    ITensorVector tensors(input_tensors);
    if (tensors.size() == 1) {
      tensors.push_back(
          network->addConstant(norm_desc->zeros.Dims(), norm_desc->zeros)->getOutput(0));
    }
    nvinfer1::IPluginV2Layer* skip =
        network->addPluginV2(tensors.data(), tensors.size(), *plugin_obj);

    if (skip == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [skip_layer_norm] layer.";
      return {};
    }

    skip->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Skip Layer Norm]")).c_str());
    return {skip->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
