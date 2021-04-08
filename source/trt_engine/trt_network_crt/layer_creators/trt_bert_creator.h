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
#include "trt_engine/trt_network_crt/plugins/emb_layer_norm_plugin/emb_layer_norm_plugin.h"
#include "trt_engine/trt_network_crt/plugins/gelu_plugin/gelu_plugin.h"
#include "trt_engine/trt_network_crt/plugins/qkv_to_context_plugin/qkv_to_context_plugin.h"
#include "trt_engine/trt_network_crt/plugins/skip_layer_norm_plugin/skip_layer_norm_plugin.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT Bert层创建器
 */
template <>
class TLayerCreator<TrtBertDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    LOG(INFO) << "TrtBertDesc::CreateLayer";
    const auto bert_desc = dynamic_cast<const TrtBertDesc*>(layer_desc);
    T_CHECK(bert_desc);

    if (bert_desc->use_fp16) {
      LOG(INFO) << "Bert : open use_fp16";
    }
    if (bert_desc->use_int8) {
      LOG(INFO) << "Bert : open use_int8";
    }
    if (bert_desc->calib_mode) {
      LOG(INFO) << "Bert : open calib_mode";
    }

    ITensorVector transposed_input_tensors;
    const nvinfer1::Permutation transpose{1, 0, 2, 3, 4, 5, 6, 7};

    for (auto* tensor : input_tensors) {
      auto shuffle_layer = network->addShuffle(*tensor);
      shuffle_layer->setFirstTranspose(transpose);
      transposed_input_tensors.push_back(shuffle_layer->getOutput(0));
    }

    nvinfer1::IPluginV2Layer* emb_layer =
        CreateEmbedLayer(network, transposed_input_tensors, bert_desc);

    if (emb_layer == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [Bert_Embed] layer.";
      return {};
    }

    /// BERT Model
    nvinfer1::ITensor* bert_output = emb_layer->getOutput(0);
    nvinfer1::ITensor* mask_idx = emb_layer->getOutput(1);

    for (int layer = 0; layer < bert_desc->n_layers; layer++) {
      std::stringstream ss;
      ss << "encoder_layer_" << layer << "_";
      // 这里做了修改！
      // ss << "l" << layer << "_";

      bert_output = CreateTransformer(ss.str(), bert_desc, network, bert_output, mask_idx);
    }

    if (bert_output == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [Transformer] layer.";
      return {};
    }

    // Reshape output for TensorRT
    nvinfer1::IShuffleLayer* reshape = AddReshape(network, bert_output);
    if (reshape == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [BERT Reshape] layer.";
      return {};
    }

    auto shuffle_layer = network->addShuffle(*reshape->getOutput(0));
    shuffle_layer->setFirstTranspose(transpose);

    return {shuffle_layer->getOutput(0)};
  }

 private:
  nvinfer1::IPluginV2Layer* CreateEmbedLayer(nvinfer1::INetworkDefinition* network,
                                             const ITensorVector& input_tensors,
                                             const TrtBertDesc* const bert_desc) const {
    const auto& beta = bert_desc->weight_map.at(NAME_MAP.at("EMB_BETA"));
    const auto& gamma = bert_desc->weight_map.at(NAME_MAP.at("EMB_GAMMA"));
    const auto& word_emb = bert_desc->weight_map.at(NAME_MAP.at("EMB_WORD"));
    const auto& tok_emb = bert_desc->weight_map.at(NAME_MAP.at("EMB_TOK"));
    const auto& pos_emb = bert_desc->weight_map.at(NAME_MAP.at("EMB_POS"));

    // if use_int8, use kFLOAT here.
    const int use_fp16 = bert_desc->use_fp16 ? 1 : 0;
    const auto mha_type =
        TrtCommon::GetDataType(bert_desc->use_fp16, bert_desc->use_int8, bert_desc->calib_mode);

    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        bert::FWD_EMB_LAYER_NORM_NAME, bert::FWD_EMB_LAYER_NORM_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back(NAME_MAP.at("EMB_BETA").c_str(), beta.Data(),
                            nvinfer1::PluginFieldType::kFLOAT32, beta.Count());
    field_data.emplace_back(NAME_MAP.at("EMB_GAMMA").c_str(), gamma.Data(),
                            nvinfer1::PluginFieldType::kFLOAT32, gamma.Count());
    field_data.emplace_back(NAME_MAP.at("EMB_WORD").c_str(), word_emb.Data(),
                            nvinfer1::PluginFieldType::kFLOAT32, word_emb.Count());
    field_data.emplace_back(NAME_MAP.at("EMB_TOK").c_str(), tok_emb.Data(),
                            nvinfer1::PluginFieldType::kFLOAT32, tok_emb.Count());
    field_data.emplace_back(NAME_MAP.at("EMB_POS").c_str(), pos_emb.Data(),
                            nvinfer1::PluginFieldType::kFLOAT32, pos_emb.Count());
    field_data.emplace_back("use_fp16", &use_fp16, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("mha_type_id", &mha_type, nvinfer1::PluginFieldType::kINT32, 1);

    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("embeddings", &plugin_data));

    nvinfer1::IPluginV2Layer* layer = network->addPluginV2(input_tensors.data(), 3, *plugin_obj);
    T_CHECK(layer);

    layer->setName("embeddings_layer");
    TrtCommon::SetOutputName(layer, "embeddings_", "output");
    return layer;
  }

  nvinfer1::ITensor* CreateTransformer(const std::string& prefix, const TrtBertDesc* bert_desc,
                                       nvinfer1::INetworkDefinition* network,
                                       nvinfer1::ITensor* input,
                                       nvinfer1::ITensor* imask = nullptr) {
    // const nvinfer1::Dims idims = input->getDimensions();
    // assert(idims.nbDims == 5);
    nvinfer1::ITensor* attention_heads =
        CreateAttentionHeads(prefix + "attention_self_", bert_desc, network, input, imask);
    T_CHECK(attention_heads);

    nvinfer1::ITensor* attention_output =
        CreateAttentionOutput(prefix, bert_desc, network, input, attention_heads);
    T_CHECK(attention_output);

    // grouped conv1d 实现方式需要对 attention output 进行 reshape
    if (bert_desc->use_group_conv1d) {
      T_CHECK_NE(bert_desc->intermediate_shape.nbDims, 0);
      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*attention_output);
      shuffle->setReshapeDimensions(bert_desc->intermediate_shape);
      attention_output = shuffle->getOutput(0);
    }

    nvinfer1::ITensor* mid_act =
        CreateIntermediateActivation(prefix, bert_desc, network, attention_output);
    T_CHECK(mid_act);

    nvinfer1::ITensor* output =
        TransformerOutput(prefix, bert_desc, network, mid_act, attention_output);

    // grouped conv1d 还原维度
    if (bert_desc->use_group_conv1d) {
      nvinfer1::Dims dims = output->getDimensions();
      T_CHECK_EQ(dims.nbDims, 3);
      dims.nbDims = 5;
      dims.d[3] = dims.d[4] = 1;
      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*output);
      shuffle->setReshapeDimensions(dims);
      output = shuffle->getOutput(0);
    }
    return output;
  }

  nvinfer1::ILayer* CreateSkipLayerNorm(const std::string& prefix, const TrtBertDesc* bert_desc,
                                        nvinfer1::INetworkDefinition* network,
                                        nvinfer1::ITensor* input, nvinfer1::ITensor* skip) {
    // const Dims idims = inputTensor->getDimensions();
    // assert(idims.nbDims == 5);
    // const int hiddenSize = idims.d[2];

    const auto& beta = bert_desc->weight_map.at(prefix + "beta");
    const auto& gamma = bert_desc->weight_map.at(prefix + "gamma");
    // 暂时没有模型使用到了bias
    // const nvinfer1::Weights& bias = bert_desc->weight_map.at(prefix +
    // "bias");

    // requiring that sm >= 72
    bool use_int8 = bert_desc->use_int8 && bert::getSMVersion() >= kSM_72;
    // requiring that hidden_size is 768 or 1024
    if (bert_desc->hidden_size != 768 && bert_desc->hidden_size != 1024) use_int8 = false;

    const auto dtype = TrtCommon::GetDataType(bert_desc->use_fp16, use_int8, bert_desc->calib_mode);

    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        bert::FWD_SKIP_LAYER_NORM_NAME, bert::FWD_SKIP_LAYER_NORM_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("beta", beta.Data(), nvinfer1::PluginFieldType::kFLOAT32, beta.Count());
    field_data.emplace_back("gamma", gamma.Data(), nvinfer1::PluginFieldType::kFLOAT32,
                            gamma.Count());
    field_data.emplace_back("ld", &bert_desc->hidden_size, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("type_id", &dtype, nvinfer1::PluginFieldType::kINT32, 1);

    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("skip_ln", &plugin_data));

    nvinfer1::ITensor* skip_ln_inputs[2] = {input, skip};

    nvinfer1::IPluginV2Layer* skip_ln_layer = network->addPluginV2(skip_ln_inputs, 2, *plugin_obj);
    skip_ln_layer->setName((prefix + "skip_layer_norm").c_str());
    TrtCommon::SetOutputName(skip_ln_layer, prefix, "output");
    return skip_ln_layer;
  }

  nvinfer1::ITensor* CreateAttentionHeads(const std::string& prefix, const TrtBertDesc* bert_desc,
                                          nvinfer1::INetworkDefinition* network,
                                          nvinfer1::ITensor* input,
                                          nvinfer1::ITensor* input_mask = nullptr) {
    const nvinfer1::Dims in_dims = input->getDimensions();

    const int hidden_size = bert_desc->hidden_size;
    const int num_heads = bert_desc->num_heads;

    T_CHECK(hidden_size % num_heads == 0);

    const auto& weights = bert_desc->weight_map.at(prefix + TrtBertDesc::WQKV);
    const auto& bias = bert_desc->weight_map.at(prefix + TrtBertDesc::BQKV);

    nvinfer1::ITensor* output;
    if (bert_desc->use_int8) {
      nvinfer1::IConvolutionLayer* qkv_mult_layer =
          network->addConvolution(*input, 3 * hidden_size, {1, 1}, weights, bias);
      T_CHECK(qkv_mult_layer);

      qkv_mult_layer->setName((prefix + "FC_QKV").c_str());
      TrtCommon::SetOutputName(qkv_mult_layer, prefix, "qkv_mult");
      output = qkv_mult_layer->getOutput(0);
    } else {
      nvinfer1::IFullyConnectedLayer* qkv_mult_layer =
          network->addFullyConnected(*input, 3 * hidden_size, weights, bias);
      T_CHECK(qkv_mult_layer);

      qkv_mult_layer->setName((prefix + "FC_QKV").c_str());
      TrtCommon::SetOutputName(qkv_mult_layer, prefix, "qkv_mult");
      output = qkv_mult_layer->getOutput(0);
    }

    const int has_mask = input_mask != nullptr ? 1 : 0;
    // here we do not use int mode for qkv layer by default, because we found
    // that it results in large errors!
    auto dtype = TrtCommon::GetDataType(bert_desc->use_fp16, bert_desc->use_int8, true);

    nvinfer1::IPluginCreator* creator = getPluginRegistry()->getPluginCreator(
        bert::FWD_QKV_TO_CONTEXT_PLUGIN_NAME, bert::FWD_QKV_TO_CONTEXT_PLUGIN_VERSION);
    std::vector<nvinfer1::PluginField> field_data;
    field_data.emplace_back("type_id", &dtype, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("hidden_size", &hidden_size, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("num_heads", &num_heads, nvinfer1::PluginFieldType::kINT32, 1);
    field_data.emplace_back("has_mask", &has_mask, nvinfer1::PluginFieldType::kINT32, 1);

    const nvinfer1::PluginFieldCollection plugin_data{static_cast<int>(field_data.size()),
                                                      field_data.data()};
    const auto plugin_obj = TrtCommon::InferUniquePtr<nvinfer1::IPluginV2>(
        creator->createPlugin("qkv2ctx", &plugin_data));

    nvinfer1::ITensor* qkv_in[2] = {output, input_mask};
    nvinfer1::IPluginV2Layer* qkv2ctx_layer =
        network->addPluginV2(qkv_in, 1 + has_mask, *plugin_obj);
    T_CHECK(qkv2ctx_layer);

    qkv2ctx_layer->setName((prefix + "QKV2CTX").c_str());
    TrtCommon::SetOutputName(qkv2ctx_layer, prefix, "context_layer");
    return qkv2ctx_layer->getOutput(0);
  }

  nvinfer1::ITensor* CreateAttentionOutput(const std::string& prefix, const TrtBertDesc* bert_desc,
                                           nvinfer1::INetworkDefinition* network,
                                           nvinfer1::ITensor* input,
                                           nvinfer1::ITensor* attention_heads) {
    const auto& weights = bert_desc->weight_map.at(prefix + NAME_MAP.at("W_AOUT"));
    const auto& bias = bert_desc->weight_map.at(prefix + NAME_MAP.at("B_AOUT"));

    nvinfer1::ITensor* output;

    if (bert_desc->use_int8) {
      nvinfer1::IConvolutionLayer* att_out_fc_layer =
          network->addConvolution(*attention_heads, bert_desc->hidden_size, {1, 1}, weights, bias);
      T_CHECK(att_out_fc_layer);

      output = att_out_fc_layer->getOutput(0);
    } else {
      nvinfer1::IFullyConnectedLayer* att_out_fc_layer =
          network->addFullyConnected(*attention_heads, bert_desc->hidden_size, weights, bias);
      T_CHECK(att_out_fc_layer);

      output = att_out_fc_layer->getOutput(0);
    }

    nvinfer1::ILayer* att_ln_layer = CreateSkipLayerNorm(prefix + "attention_output_layernorm_",
                                                         bert_desc, network, output, input);
    T_CHECK(att_ln_layer);

    return att_ln_layer->getOutput(0);
  }

  nvinfer1::ITensor* CreateIntermediateActivation(const std::string& prefix,
                                                  const TrtBertDesc* bert_desc,
                                                  nvinfer1::INetworkDefinition* network,
                                                  nvinfer1::ITensor* attention_outputs) {
    nvinfer1::ITensor* default_return_value = nullptr;

    // grouped conv1d or dense
    nvinfer1::ITensor* mid_act = nullptr;

    if (bert_desc->use_group_conv1d) {
      mid_act =
          CreateGroupedConv1D(prefix + "intermediate_", bert_desc, network, attention_outputs);
      TrtCommon::SetTensorName(mid_act, prefix + "intermediate_", "concat");
    } else {
      const nvinfer1::Weights weights = bert_desc->weight_map.at(prefix + NAME_MAP.at("W_MID"));
      const nvinfer1::Weights bias = bert_desc->weight_map.at(prefix + NAME_MAP.at("B_MID"));

      const int intermediate_size = bert_desc->hidden_size * 4;

      nvinfer1::ITensor* mid_out;
      if (bert_desc->use_int8) {
        nvinfer1::IConvolutionLayer* mid_dense_layer =
            network->addConvolution(*attention_outputs, intermediate_size, {1, 1}, weights, bias);
        T_CHECK(mid_dense_layer);
        mid_out = mid_dense_layer->getOutput(0);
      } else {
        nvinfer1::IFullyConnectedLayer* mid_dense_layer =
            network->addFullyConnected(*attention_outputs, intermediate_size, weights, bias);
        T_CHECK(mid_dense_layer);
        mid_out = mid_dense_layer->getOutput(0);
      }
      // gelu or relu
      if (bert_desc->use_relu) {
        nvinfer1::IActivationLayer* relu_layer =
            network->addActivation(*mid_out, nvinfer1::ActivationType::kRELU);
        T_CHECK(relu_layer);

        // reluLayer->setName("relu");
        mid_act = relu_layer->getOutput(0);
      } else {
        mid_act = bert::CreateGeluLayer(network, mid_out, bert_desc->use_fp16, bert_desc->use_int8);
      }

      T_CHECK(mid_act);
      TrtCommon::SetTensorName(mid_act, prefix, bert_desc->use_relu ? "relu" : "gelu");
    }

    return mid_act;
  }

  nvinfer1::ITensor* TransformerOutput(const std::string& prefix, const TrtBertDesc* bert_desc,
                                       nvinfer1::INetworkDefinition* network,
                                       nvinfer1::ITensor* mid_act,
                                       nvinfer1::ITensor* attention_output) {
    nvinfer1::ITensor* dense_output{nullptr};
    if (bert_desc->use_group_conv1d) {
      dense_output = CreateGroupedConv1D(prefix + "output_", bert_desc, network, mid_act);
      TrtCommon::SetTensorName(mid_act, prefix + "output_", "concat");
    } else {
      // dense to hidden size
      const nvinfer1::Weights weights = bert_desc->weight_map.at(prefix + NAME_MAP.at("W_LOUT"));
      const nvinfer1::Weights bias = bert_desc->weight_map.at(prefix + NAME_MAP.at("B_LOUT"));

      if (bert_desc->use_int8) {
        nvinfer1::IConvolutionLayer* out_dense_layer =
            network->addConvolution(*mid_act, bert_desc->hidden_size, {1, 1}, weights, bias);
        T_CHECK(out_dense_layer);
        TrtCommon::SetOutputName(out_dense_layer, prefix + "output_", "dense");

        dense_output = out_dense_layer->getOutput(0);
      } else {
        nvinfer1::IFullyConnectedLayer* out_dense_layer =
            network->addFullyConnected(*mid_act, bert_desc->hidden_size, weights, bias);
        T_CHECK(out_dense_layer);
        TrtCommon::SetOutputName(out_dense_layer, prefix + "output_", "dense");

        dense_output = out_dense_layer->getOutput(0);
      }
    }

    nvinfer1::ILayer* out_ln_layer = CreateSkipLayerNorm(prefix + "output_layernorm_", bert_desc,
                                                         network, dense_output, attention_output);
    T_CHECK(out_ln_layer);
    return out_ln_layer->getOutput(0);
  }

  nvinfer1::ITensor* CreateGroupedConv1D(const std::string& prefix, const TrtBertDesc* bert_desc,
                                         nvinfer1::INetworkDefinition* network,
                                         nvinfer1::ITensor* input) const {
    LOG(INFO) << "TrtBertDesc::CreateGroupedConv1D for prefix " << prefix;

    // 1. split layer, 默认在最后一个维度上做切分
    nvinfer1::Dims input_dims = input->getDimensions();
    nvinfer1::Dims start, stride, size;
    start.nbDims = stride.nbDims = size.nbDims = input_dims.nbDims;
    for (int i = 0; i < input_dims.nbDims; ++i) {
      start.d[i] = 0;
      stride.d[i] = 1;
      size.d[i] = input_dims.d[i];
    }
    const int num_split = bert_desc->num_split;
    const int chunk_size = input_dims.d[input_dims.nbDims - 1] / num_split;
    size.d[input_dims.nbDims - 1] = chunk_size;

    ITensorVector tensors;
    for (int i = 0; i < num_split; ++i) {
      tensors.push_back(network->addSlice(*input, start, size, stride)->getOutput(0));
      start.d[input_dims.nbDims - 1] += chunk_size;
    }

    // 2. expand dims & permute
    T_CHECK_EQ(input_dims.nbDims, 3);
    nvinfer1::Dims new_dims = tensors[0]->getDimensions();
    for (int i = new_dims.nbDims - 1; i > 0; --i) {
      new_dims.d[i + 1] = new_dims.d[i];
    }
    new_dims.d[1] = 1;
    ++new_dims.nbDims;
    for (int i = 0; i < tensors.size(); ++i) {
      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*tensors[i]);
      shuffle->setReshapeDimensions(new_dims);
      shuffle->setSecondTranspose({0, 3, 1, 2});  // NHWC->NCHW
      tensors[i] = shuffle->getOutput(0);
    }

    // 3. conv 1d
    for (int i = 0; i < tensors.size(); ++i) {
      const FwdWeights& weights =
          bert_desc->weight_map.at(prefix + std::string("grouped_convolution_") +
                                   std::to_string(i) + std::string("_conv1d_expanddims_1"));
      const FwdWeights& bias =
          bert_desc->weight_map.at(prefix + std::string("grouped_convolution_") +
                                   std::to_string(i) + std::string("_bias_read"));

      const auto filter_dims = weights.Dims();
      const nvinfer1::Dims kernelSize =
          TrtUtils::ToDims(std::vector<int>{filter_dims.d[2], filter_dims.d[3]});
      const int nbOutputMaps = filter_dims.d[0];
      nvinfer1::IConvolutionLayer* conv =
          network->addConvolutionNd(*tensors[i], nbOutputMaps, kernelSize, weights, bias);

      // TODO(Ao Li): 这里固定这些值
      conv->setNbGroups(1);
      conv->setStrideNd({2, {1, 1}});
      conv->setDilationNd({2, {1, 1}});
      conv->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

      tensors[i] = conv->getOutput(0);
    }

    // 4. squeeze
    for (int i = 0; i < tensors.size(); ++i) {
      // TODO(Ao Li): 认为 squeeze dim 是 1
      nvinfer1::Dims dim = tensors[i]->getDimensions();
      T_CHECK_EQ(dim.nbDims, 4);
      dim.d[2] = dim.d[1];
      dim.d[1] = dim.d[3];
      --dim.nbDims;
      nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*tensors[i]);
      shuffle->setFirstTranspose({0, 2, 3, 1});  // NCHW->NHWC
      shuffle->setReshapeDimensions(dim);
      tensors[i] = shuffle->getOutput(0);
    }

    // 5. concatenate TODO(Ao Li)： 默认在最后一个维度上做连接
    nvinfer1::IConcatenationLayer* concat =
        network->addConcatenation(tensors.data(), tensors.size());
    concat->setAxis(tensors[0]->getDimensions().nbDims - 1);
    return concat->getOutput(0);
  }

  nvinfer1::IShuffleLayer* AddReshape(nvinfer1::INetworkDefinition* network,
                                      nvinfer1::ITensor* bert_output) {
    auto output_dims = bert_output->getDimensions();
    output_dims.nbDims = 3;
    nvinfer1::IShuffleLayer* reshape = network->addShuffle(*bert_output);
    reshape->setReshapeDimensions(output_dims);
    // Permutation perm{1, 0, 2, 0, 0, 0, 0, 0};
    // reshape->setSecondTranspose(perm);
    return reshape;
  }

  const std::unordered_map<std::string, std::string> NAME_MAP{
      {"EMB_BETA", "embedding_layernorm_beta"},
      {"EMB_GAMMA", "embedding_layernorm_gamma"},
      {"EMB_WORD", "embedding_word_embedding"},
      {"EMB_TOK", "embedding_token_type_embedding"},
      {"EMB_POS", "embedding_position_embedding"},

      {"W_AOUT", "attention_output_dense_kernel"},
      {"B_AOUT", "attention_output_dense_bias"},
      {"AOUT_LN_BETA", "attention_output_layernorm_beta"},
      {"AOUT_LN_GAMMA", "attention_output_layernorm_gamma"},
      {"W_MID", "intermediate_dense_kernel"},
      {"B_MID", "intermediate_dense_bias"},
      {"W_LOUT", "output_dense_kernel"},
      {"B_LOUT", "output_dense_bias"},
      {"LOUT_LN_BETA", "output_layernorm_beta"},
      {"LOUT_LN_GAMMA", "output_layernorm_gamma"},
  };
};

FWD_TRT_NAMESPACE_END
