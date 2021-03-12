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

#include "fwd_keras/keras_cvt/trt_keras_parser.h"

#include <memory>
#include <string>
#include <vector>

#include "keras_desc_creators/keras_activation_creator.h"
#include "keras_desc_creators/keras_batch_norm_creator.h"
#include "keras_desc_creators/keras_clamp_creator.h"
#include "keras_desc_creators/keras_concatenation_creator.h"
#include "keras_desc_creators/keras_convolution_creator.h"
#include "keras_desc_creators/keras_dense_creator.h"
#include "keras_desc_creators/keras_element_wise_creator.h"
#include "keras_desc_creators/keras_gather_creator.h"
#include "keras_desc_creators/keras_noop_creator.h"
#include "keras_desc_creators/keras_pad_creator.h"
#include "keras_desc_creators/keras_pooling_creator.h"
#include "keras_desc_creators/keras_reduce_creator.h"
#include "keras_desc_creators/keras_rnn_creator.h"
#include "keras_desc_creators/keras_separable_conv_creator.h"
#include "keras_desc_creators/keras_shuffle_creator.h"
#include "keras_desc_creators/keras_slice_creator.h"
#include "keras_desc_creators/keras_softmax_creator.h"

#ifdef _MSC_VER
#include <windows.h>
#endif  // _MSC_VER

FWD_KERAS_NAMESPACE_BEGIN
Parser::Parser(InferMode mode) : mode_(mode) {
  network_.outputs.clear();
  network_.inputs.clear();

  // 这里注册的时候，将复杂的描述放在上面，会先检测复杂的模式0
  RegisterCreator<TrtRNNv2Desc>();
  RegisterCreator<TrtActivationDesc>();
  RegisterCreator<TrtConvolutionDesc>();
  RegisterCreator<TrtPoolingDesc>();
  RegisterCreator<TrtNormalizationDesc>();
  RegisterCreator<TrtFullyConnectedDesc>();
  RegisterCreator<TrtSoftmaxDesc>();
  RegisterCreator<TrtSeparableConvDesc>();
  RegisterCreator<TrtConcatenationDesc>();
  RegisterCreator<TrtConstantPadDesc>();
  RegisterCreator<TrtShuffleDesc>();
  RegisterCreator<TrtReduceDesc>();
  RegisterCreator<TrtSliceDesc>();
  RegisterCreator<TrtElementWiseDesc>();
  RegisterCreator<TrtClampDesc>();

  RegisterCreator<TrtGatherDesc>();
  // 这种简单的模式就放在下面
  RegisterCreator<TrtNoopDesc>();
}

bool Parser::Parse(const std::string& graph_path, int batch_size) {
  value_to_layer_descs_.clear();

  network_.batch_size = batch_size;

  if (!model_reader_.LoadModel(graph_path)) {
    LOG(ERROR) << "Load Model failed.";
    return false;
  }

  if (!CreateInputDescs(model_reader_.Inputs(), batch_size)) {
    LOG(ERROR) << "Create Input Failed.";
    return false;
  }

  for (auto* graph_output : model_reader_.Outputs()) {
    auto output_desc = std::make_shared<TrtOutputDesc>();
    if (!ParseOperaion(output_desc.get(), *graph_output)) {
      return false;
    }

    ProcessOutputDesc(*graph_output, output_desc.get());

    network_.outputs.push_back(output_desc);
  }

  return true;
}

const TrtNetworkDesc& Parser::GetNetwork() const { return network_; }

bool Parser::CreateInputDescs(const std::vector<const Layer*>& inputs, int batch_size) {
  bool default_return_value = false;

  for (auto* input_ptr : inputs) {
    const Layer& input = *input_ptr;

    auto input_desc = std::make_shared<TrtInputDesc>();

    input_desc->name = input.Name();
    const auto dims = input.GetAttr<json>("batch_input_shape");
    const std::string d_type = input.GetAttr<std::string>("dtype");

    if (d_type == "int32") {
      input_desc->type = nvinfer1::DataType::kINT32;
    } else {
      T_CHECK(d_type == "float32");
      input_desc->type = nvinfer1::DataType::kFLOAT;
    }

    input_desc->dimensions = Utils::DimsOf(dims);

    // TODO(Ao Li): 目前将 batch_size 根据传入参数固定，以后可以尝试改为动态
    if (input_desc->dimensions.d[0] == -1) {
      input_desc->dimensions.d[0] = batch_size;
    }

    network_.inputs.push_back(input_desc);

    ProcessInputDesc(input, input_desc);

    // LOG(INFO) << "Input = " << (input) << " : " << input_desc->name;
  }

  return true;
}

void Parser::ProcessInputDesc(const Layer& input, std::shared_ptr<TrtInputDesc> input_desc) {
  // TODO(Ao Li): 这里会将输入维度为 4 的情况视为 NHWC，自动添加 Shuffle
  // 层用于输入格式转换
  if (input_desc->dimensions.nbDims == 4) {
    LOG(INFO) << "Parser::CreateNHWC2NCHWLayerDesc";
    auto shuffle_desc = CreateNHWC2NCHWLayerDesc();
    shuffle_desc->inputs.push_back({input_desc, 0});
    value_to_layer_descs_.insert({input.get(), shuffle_desc});
  } else {
    value_to_layer_descs_.insert({input.get(), input_desc});
  }
}

void Parser::ProcessOutputDesc(const Layer& output_layer, TrtOutputDesc* output_desc) {
  // TODO(Ao Li): 如果输出维度是 4，在此处将输出从 NCHW 转换为 NHWC 格式
  // TODO(Ao Li): 目前无法获取输出维度，暂时用 DescName 做判断，之后可能需要传个
  // dummy_output_shape
  std::string layer_type = output_layer.Type();

  if (layer_type == "Dense" || layer_type == "Embedding" || layer_type == "SimpleRNN" ||
      layer_type == "GRU" || layer_type == "Bidirectional" || layer_type == "LSTM")
    return;

  LOG(INFO) << "Parser::CreateNCHW2NHWCLayerDesc";
  auto shuffle_desc = CreateNCHW2NHWCLayerDesc();
  shuffle_desc->inputs = output_desc->inputs;
  output_desc->inputs = {{shuffle_desc, 0}};
}

bool Parser::ParseOperaion(TrtLayerDesc* parent, const Layer& layer) {
  const auto iter = value_to_layer_descs_.find(layer.get());

  // 已经创建好的，直接加入到输入中
  if (iter != value_to_layer_descs_.end()) {
    parent->inputs.push_back({iter->second, 0});
    return true;
  }

  auto layer_creator = FindDescCreator(layer);
  if (layer_creator == nullptr) return false;

  std::vector<std::string> input_values;
  const auto layer_desc = layer_creator->Create(layer, model_reader_, input_values);

  if (layer_desc == nullptr || input_values.empty()) {
    const auto name = "";  // keras_OperationName(op);
#ifdef _MSC_VER
    char text[256];
    sprintf_s(text, 256, "Creating %s Desc failed! Please Check implementation and inputs.", name);
    ::MessageBox(NULL, text, "Error", MB_OK | MB_ICONERROR);
#endif  // _MSC_VER
    LOG(ERROR) << "Creating " << name << "Desc failed! Please Check implementation and inputs.";
    return false;
  }

  parent->inputs.push_back({layer_desc, 0});
  value_to_layer_descs_.insert({layer.get(), layer_desc});

  for (const auto& input_value : input_values) {
    if (!ParseOperaion(layer_desc.get(), model_reader_.GetLayer(input_value))) {
      return false;
    }
  }

  return true;
}

ILayerDescCreator* Parser::FindDescCreator(const Layer& layer) {
  // 查找对应的层描述创建器
  for (const auto& creator : layer_desc_creators_) {
    if (creator->Check(layer)) {
      return creator.get();
    }
  }

  const auto name = "";  // keras_OperationName(op);
#ifdef _MSC_VER
  char text[256];
  sprintf_s(text, 256, "Cannot found %s Creator!", name);
  ::MessageBox(NULL, text, "Error", MB_OK | MB_ICONERROR);
#endif  // _MSC_VER
  LOG(ERROR) << "Could not find layer create for operation " << name;

  return nullptr;
}

std::shared_ptr<TrtShuffleDesc> Parser::CreateNHWC2NCHWLayerDesc() {
  auto shuffle_desc = std::make_shared<TrtShuffleDesc>();
  shuffle_desc->doFirstTrans = true;
  shuffle_desc->doReshape = false;
  shuffle_desc->doSecondTrans = false;
  shuffle_desc->firstTranspose = {0, 3, 1, 2, 0, 0, 0, 0};
  return shuffle_desc;
}

std::shared_ptr<TrtShuffleDesc> Parser::CreateNCHW2NHWCLayerDesc() {
  auto shuffle_desc = std::make_shared<TrtShuffleDesc>();
  shuffle_desc->doFirstTrans = true;
  shuffle_desc->doReshape = false;
  shuffle_desc->doSecondTrans = false;
  shuffle_desc->firstTranspose = {0, 2, 3, 1, 0, 0, 0, 0};
  return shuffle_desc;
}

FWD_KERAS_NAMESPACE_END
