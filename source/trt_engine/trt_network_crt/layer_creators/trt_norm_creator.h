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
#include "trt_engine/trt_network_crt/plugins/norm_plugin/norm_plugin.h"
#include "trt_engine/trt_network_crt/plugins/reduce_plugin/reduce_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT 层压缩层创建器
 */
template <>
class TLayerCreator<TrtNormDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtNormDesc::CreatePlugin";
    const auto norm_desc = dynamic_cast<const TrtNormDesc*>(layer_desc);
    T_CHECK(norm_desc);

    nvinfer1::ITensor* input = input_tensors[0];

    const auto reduce_dims = CalcReduceDims(norm_desc->axes);
    LOG(INFO) << "reduce_dims = " << TrtUtils::ValueStrOf(reduce_dims);

    // reduce
    // TODO(Paul Lu): reduce 和 norm 存在一处不一致未处理：当power=1时 norm
    // 需要取绝对值
    nvinfer1::IPluginCreator* creator = nullptr;
    if (norm_desc->div) {
      creator = getPluginRegistry()->getPluginCreator(NORM_PLUGIN_NAME, NORM_PLUGIN_VERSION);
    } else {
      creator = getPluginRegistry()->getPluginCreator(REDUCE_PLUGIN_NAME, REDUCE_PLUGIN_VERSION);
    }
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("reduce_dims", reduce_dims.data(), nvinfer1::PluginFieldType::kINT32,
                            reduce_dims.size());
    field_data.emplace_back("keep_dim", &norm_desc->keepDim, nvinfer1::PluginFieldType::kINT32, 1);

    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("power", &norm_desc->p, nvinfer1::PluginFieldType::kFLOAT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};

    const auto plugin_obj =
        TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("norm", &plugin_data));

    // add the plugin to the TensorRT network
    const auto norm = network->addPluginV2(&input, 1, *plugin_obj);

    if (norm == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [Norm] layer.";
      return {};
    }

    norm->setName((std::to_string(network->getNbLayers()) + std::string(" [Norm Plugin]")).c_str());
    ITensorVector outputs;
    for (int i = 0; i < norm->getNbOutputs(); ++i) {
      outputs.push_back(norm->getOutput(i));
    }
    return outputs;
  }

 private:
  static std::vector<int> CalcReduceDims(uint32_t reduceAxes) {
    std::vector<int> reduce_dims;
    for (int i = 0; i < nvinfer1::Dims::MAX_DIMS; ++i) {
      if (reduceAxes & (1 << i)) {
        reduce_dims.push_back(i);
      }
    }
    return reduce_dims;
  }
};

FWD_TRT_NAMESPACE_END
