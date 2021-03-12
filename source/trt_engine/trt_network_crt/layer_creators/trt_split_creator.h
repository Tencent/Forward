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
#include "trt_engine/trt_network_crt/plugins/split_plugin/split_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT 切片层创建器
 */
template <>
class TLayerCreator<TrtSplitDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    // #define USE_SPLIT_PLUGIN

#ifdef USE_SPLIT_PLUGIN
    return CreatePlugin(network, layer_desc, input_tensors);
#else
    LOG(INFO) << "TrtSplitDesc::CreateLayer";
    const auto split_desc = dynamic_cast<const TrtSplitDesc*>(layer_desc);
    T_CHECK(split_desc);

    auto& input = *input_tensors[0];
    const int dim = split_desc->dim;
    const auto& split_sizes = split_desc->splitSize;
    const auto& chunk_sizes = split_desc->chunk_sizes;
    const auto& dynamic_size = split_desc->dynamic_size;
    ITensorVector outputs;

    int start_offset = 0;
    for (int i = 0; i < split_sizes.size(); ++i) {
      nvinfer1::ITensor* tensor = CreateSliceLayer(network, input, dim, start_offset,
                                                   split_sizes[i], chunk_sizes[i], dynamic_size);
      if (tensor == nullptr) {
        LOG(ERROR) << "Create Network: Fail to create [split::slice] layer.";
        return {};
      }

      outputs.push_back(tensor);

      start_offset += split_sizes[i];
    }

    return outputs;
#endif  // USE_SPLIT_PLUGIN
  }

  ITensorVector CreatePlugin(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                             const ITensorVector& input_tensors) const {
    LOG(INFO) << "TrtSplitDesc::CreatePlugin";
    const auto split_desc = dynamic_cast<const TrtSplitDesc*>(layer_desc);
    T_CHECK(split_desc);

    auto& input = *input_tensors[0];
    const int dim = split_desc->dim;
    auto& split_sizes = split_desc->splitSize;

    nvinfer1::IPluginCreator* creator =
        getPluginRegistry()->getPluginCreator(SPLIT_PLUGIN_NAME, SPLIT_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("split_size", split_desc->splitSize.data(),
                            nvinfer1::PluginFieldType::kINT32, split_desc->splitSize.size());
    field_data.emplace_back("dim", &split_desc->dim, nvinfer1::PluginFieldType::kINT32, 1);

    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("split", &plugin_data));

    // add the plugin to the TensorRT network
    const auto split = network->addPluginV2(input_tensors.data(), 1, *plugin_obj);

    if (split == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [split] layer.";
      return {};
    }

    split->setName((std::to_string(network->getNbLayers()) + std::string(" [Split]")).c_str());
    ITensorVector outputs;
    for (int i = 0; i < split->getNbOutputs(); ++i) {
      outputs.push_back(split->getOutput(i));
    }
    return outputs;
  }

  nvinfer1::ITensor* CreateSliceLayer(nvinfer1::INetworkDefinition* network,
                                      nvinfer1::ITensor& input, int dim, int start, int size,
                                      const FwdWeights& chunk_size, bool dynamic_size) {
    auto dims = input.getDimensions();
    TrtSliceDesc slice_desc;
    // 初始化
    slice_desc.start.nbDims = dims.nbDims;
    slice_desc.stride.nbDims = dims.nbDims;
    slice_desc.size.nbDims = dims.nbDims;
    for (int i = 0; i < dims.nbDims; ++i) {
      slice_desc.start.d[i] = 0;
      slice_desc.stride.d[i] = 1;
      slice_desc.size.d[i] = dims.d[i];
      if (dims.d[i] < 0) dynamic_size = true;
    }

    // 对目标 dim 进行 slice
    slice_desc.start.d[dim] = start;
    slice_desc.size.d[dim] = size;

    nvinfer1::ISliceLayer* slice =
        network->addSlice(input, slice_desc.start, slice_desc.size, slice_desc.stride);

    T_CHECK(slice);

    if (dynamic_size) {
      auto shape = network->addShape(input)->getOutput(0);
      T_CHECK(shape);

      auto chunk_tmp = network->addConstant({1, dims.nbDims}, chunk_size)->getOutput(0);
      auto chunk = network->addElementWise(*chunk_tmp, *shape, nvinfer1::ElementWiseOperation::kMIN)
                       ->getOutput(0);

      slice->setInput(2, *chunk);
    }

    return slice->getOutput(0);
  }
};

FWD_TRT_NAMESPACE_END
