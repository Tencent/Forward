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

#include "fwd_tf/tf_cvt/tf_graph_parser.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common/fwd_common.h"

FWD_TF_NAMESPACE_BEGIN

Parser::Parser(InferMode mode) : mode_(mode) {}

bool Parser::Parse(const std::string& graph_path,
                   const std::unordered_map<std::string, TF_Tensor*>& dummy_input_map) {
  if (!graph_.Load(graph_path, mode_)) {
    LOG(ERROR) << "Opening graph file is failed.";
    return false;
  }

  if (!CreateInputDescs(graph_.Inputs(), dummy_input_map)) {
    LOG(ERROR) << "Creating input desc is failed.";
    return false;
  }

  if (!SetNetworkBatchSize()) return false;

  for (auto graph_output : graph_.Outputs()) {
    auto output_desc = std::make_shared<TrtOutputDesc>();
    // 默认每个output节点只有一个输出
    if (!ParseOperaion(output_desc.get(), graph_output, 0)) {
      return false;
    }
    ProcessOutputDesc(graph_output, output_desc.get());

    network_.outputs.push_back(output_desc);
  }

  return true;
}

const TrtNetworkDesc& Parser::GetNetwork() const { return network_; }

std::set<int> Parser::GetUnusedInputs() const { return network_.unused_input_indices; }

bool Parser::SetInputType(const Operation& input, std::shared_ptr<TrtInputDesc> input_desc,
                          const TF_Tensor* dummy_input) const {
  const auto input_type = TF_TensorType(dummy_input);

  switch (input_type) {
    case TF_DataType::TF_FLOAT:
    case TF_DataType::TF_HALF:
      input_desc->type =
          mode_ == InferMode::FLOAT ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
      break;
    case TF_DataType::TF_INT64:
    case TF_DataType::TF_INT32:
      input_desc->type = nvinfer1::DataType::kINT32;
      break;
    default:
      LOG(ERROR) << "Unsupported Input type for Input = " << input.Name() << " : "
                 << input_desc->name;
      return false;
  }
  return true;
}

bool Parser::CreateInputDescs(const std::vector<Operation>& inputs,
                              const std::unordered_map<std::string, TF_Tensor*>& dummy_input_map) {
  // TODO(yzx): 暂时取消处理 IteratorGetNext
  // 输入，推荐用脚本进行模型输入重置
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Operation& input = inputs[i];

    auto input_desc = std::make_shared<TrtInputDesc>();

    const auto& input_entry = dummy_input_map.find(input.Name());
    if (input_entry == dummy_input_map.end()) {
      return false;
    }

    input_desc->name = input_entry->first;
    if (!SetInputType(input, input_desc, input_entry->second)) return false;
    input_desc->dimensions = Utils::DimsOf(input_entry->second);

    network_.inputs.push_back(input_desc);

    ProcessInputDesc(input, input_desc);

    LOG(INFO) << "Input = " << input.Name() << " : " << input_desc->name;
  }

  if (network_.inputs.empty()) {
    LOG(ERROR) << "No input desc has been created.";
    return false;
  }
  return true;
}

bool Parser::ParseOperaion(TrtLayerDesc* parent, const Operation& op, int index) {
  const auto iter = created_desc_map_.find(op.Op());

  // 已经创建好的，直接加入到输入中
  if (iter != created_desc_map_.end()) {
    parent->inputs.push_back({iter->second, index});
    return true;
  }

  auto layer_creator = desc_manager_.FindDescCreator(op);
  if (layer_creator == nullptr) return false;

  std::vector<Output> input_values;
  const auto layer_desc = layer_creator->Create(op, graph_, input_values);

  if (layer_desc == nullptr || input_values.empty()) {
    LOG(ERROR) << "Creating " << op.Name()
               << "Desc failed! Please Check implementation and inputs.";
    return false;
  }

  parent->inputs.push_back({layer_desc, index});
  created_desc_map_[op.Op()] = layer_desc;

  for (auto input_value : input_values) {
    if (!input_value.Op()) continue;

    if (!ParseOperaion(layer_desc.get(), input_value, input_value.Index())) {
      return false;
    }
  }

  return true;
}

void Parser::ProcessInputDesc(const Operation& input, std::shared_ptr<TrtInputDesc> input_desc) {
  // TODO(Ao Li): 这里会将输入维度为 4 的情况视为 NHWC，自动添加 Shuffle
  // 层用于输入格式转换
  if (input_desc->dimensions.nbDims == 4) {
    LOG(INFO) << "Parser::CreateNHWC2NCHWLayerDesc";
    auto shuffle_desc = CreateNHWC2NCHWLayerDesc();
    shuffle_desc->inputs.push_back({input_desc, 0});
    created_desc_map_[input.Op()] = shuffle_desc;
  } else {
    created_desc_map_[input.Op()] = input_desc;
  }
}

void Parser::ProcessOutputDesc(const Operation& graph_output, TrtOutputDesc* output_desc) const {
  // TODO(Ao Li): 如果输出维度是 4，在此处将输出从 NCHW 转换为 NHWC 格式
  if (Utils::DimsOf(graph_output).nbDims == 4) {
    LOG(INFO) << "Parser::CreateNCHW2NHWCLayerDesc";
    auto shuffle_desc = CreateNCHW2NHWCLayerDesc();
    shuffle_desc->inputs = output_desc->inputs;
    output_desc->inputs = {{shuffle_desc, 0}};
  }
  output_desc->name = graph_output.Name();
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

bool Parser::SetNetworkBatchSize() {
  const int batch_size = network_.inputs[0]->dimensions.d[0];
  for (auto& input : network_.inputs) {
    if (batch_size != input->dimensions.d[0]) {
      LOG(ERROR) << "Batch sizes of inputs are not consistent! Please check "
                    "inputs' dimensions.";
      return false;
    }

#ifdef USE_DYNAMIC_BATCH
    input->dimensions.d[0] = -1;
#endif  // USE_DYNAMIC_BATCH
  }

  network_.batch_size = batch_size;
  return true;
}

FWD_TF_NAMESPACE_END
