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
#include "trt_engine/trt_network_crt/plugins/reduce_plugin/reduce_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT 层压缩层创建器
 */
template <>
class TLayerCreator<TrtReduceDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtReduceDesc::CreateLayer";
    const auto reduce_desc = dynamic_cast<const TrtReduceDesc*>(layer_desc);
    T_CHECK(reduce_desc);

    auto& input = *input_tensors[0];
    uint32_t reduce_axes = reduce_desc->reduceAxes;
    if (reduce_axes == 0) {
      reduce_axes = (1 << input.getDimensions().nbDims) - 1;
    } else if (reduce_axes >= (1 << input.getDimensions().nbDims)) {
      reduce_axes = 1 << (input.getDimensions().nbDims - 1);
    }

    bool keep_dim = reduce_desc->keepDimensions;

    // 计算方差
    if (reduce_desc->isVarOp || reduce_desc->isStdOp) {
      auto mean = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, reduce_axes, true)
                      ->getOutput(0);
      auto var =
          network->addElementWise(input, *mean, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
      var =
          network->addElementWise(*var, *var, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
      var = network->addReduce(*var, nvinfer1::ReduceOperation::kAVG, reduce_axes, keep_dim)
                ->getOutput(0);

      // 修正无偏
      if (reduce_desc->unbiased) {
        auto bias = network
                        ->addConstant(TrtUtils::CreateSingleton(var->getDimensions().nbDims),
                                      {nvinfer1::DataType::kFLOAT, &reduce_desc->bias, 1})
                        ->getOutput(0);
        var = network->addElementWise(*var, *bias, nvinfer1::ElementWiseOperation::kPROD)
                  ->getOutput(0);
      }

      if (reduce_desc->isStdOp) {
        var = network->addUnary(*var, nvinfer1::UnaryOperation::kSQRT)->getOutput(0);
      }
      return {var};
    }

#define USE_REDUCE_PLUGIN

#ifdef USE_REDUCE_PLUGIN
    // TODO(Ao Li): 暂时只在 reduce_dims.size() == 1 时的 fp32 SUM op 启用 plugin
    if (reduce_desc->operation == nvinfer1::ReduceOperation::kSUM &&
        network->getInput(0)->getType() == nvinfer1::DataType::kFLOAT) {
      const auto reduce_dims = CalcReduceDims(reduce_desc->reduceAxes);
      if (reduce_dims.size() == 1) {
        return CreatePlugin(network, layer_desc, input_tensors);
      }
    }
#endif

    nvinfer1::IReduceLayer* reduce =
        network->addReduce(input, reduce_desc->operation, reduce_axes, keep_dim);

    if (reduce == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [reduce] layer.";
      return {};
    }

    return {reduce->getOutput(0)};
  }

 private:
  ITensorVector CreatePlugin(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                             const ITensorVector& input_tensors) const {
    LOG(INFO) << "TrtReduceDesc::CreatePlugin";
    const auto reduce_desc = dynamic_cast<const TrtReduceDesc*>(layer_desc);
    T_CHECK(reduce_desc);

    nvinfer1::ITensor* input = input_tensors[0];

    // 通过 shuffle 解决全局内存访问不连续的问题，需要评估这种实现的效率
    const auto input_dim = input->getDimensions();
    const auto reduce_dims = CalcReduceDims(reduce_desc->reduceAxes);
    LOG(INFO) << "reduce_dims = " << TrtUtils::ValueStrOf(reduce_dims);

#ifdef SHUFFLE_REDUCE
    if (reduce_dims.size() != 1 || reduce_dims[0] != input_dim.nbDims - 1) {
      Permutation perm{};
      int k = 0;
      for (int i = 0; i < input_dim.nbDims; ++i) {
        if (std::find(reduce_dims.begin(), reduce_dims.end(), i) == reduce_dims.end()) {
          perm.order[k++] = i;
        }
      }
      for (auto d : reduce_dims) {
        perm.order[k++] = d;
      }
      LOG(INFO) << "Shuffle in reduce plugin, before: " << TrtUtils::ShapeStrOf(input_dim);
      auto shuffle = network->addShuffle(*input);
      shuffle->setFirstTranspose(perm);
      input = shuffle->getOutput(0);
      LOG(INFO) << "Perm order: "
                << TrtUtils::ValueStrOf(std::vector<int>{perm.order, perm.order + k});
      LOG(INFO) << "End shuffle: " << TrtUtils::ShapeStrOf(input->getDimensions());
    }
#endif

    // reduce
    nvinfer1::IPluginCreator* creator =
        getPluginRegistry()->getPluginCreator(REDUCE_PLUGIN_NAME, REDUCE_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("reduce_dims", reduce_dims.data(), nvinfer1::PluginFieldType::kINT32,
                            reduce_dims.size());
    field_data.emplace_back("keep_dim", &reduce_desc->keepDimensions,
                            nvinfer1::PluginFieldType::kINT32, 1);

    nvinfer1::DataType input_type = network->getInput(0)->getType();
    field_data.emplace_back("data_type", &input_type, nvinfer1::PluginFieldType::kINT32, 1);

    // fill data
    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("reduce", &plugin_data));

    // add the plugin to the TensorRT network
    const auto reduce = network->addPluginV2(&input, 1, *plugin_obj);

    if (reduce == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [reduce] layer.";
      return {};
    }

    reduce->setName(
        (std::to_string(network->getNbLayers()) + std::string(" [Reduce Plugin]")).c_str());
    ITensorVector outputs;
    for (int i = 0; i < reduce->getNbOutputs(); ++i) {
      outputs.push_back(reduce->getOutput(i));
    }
    return outputs;
  }

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
