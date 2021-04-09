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

#include "fwd_torch/torch_cvt/torch_module.h"

#include <torch/csrc/jit/passes/fuse_linear.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if FWD_TORCH_VERSION > 160
#include <torch/csrc/jit/passes/inliner.h>

#include "torch_passes/fold_floor_divide.h"
#endif

#include "torch_passes/fuse_adaptive_lin.h"
#include "torch_passes/fuse_lrn.h"
#include "torch_passes/fuse_transformer_encoder.h"

// 开启这个选项会注册 torch 自定义 op 来改变 torch graph, 便于识别
#define USE_TORCH_CUSTOM_OPS

#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

inline void RemoveRedundantNodesOnBlock(torch::jit::Block* block) {
  auto nodes = block->nodes();
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    for (auto& sub : it->blocks()) {
      RemoveRedundantNodesOnBlock(sub);
    }

    const auto& kind = it->kind();
    // TODO(Ao Li): add more redundant nodes here
    if (kind == c10::aten::to || kind == c10::aten::device || kind == c10::aten::detach ||
        kind == c10::aten::contiguous) {
      if (it->schema().overload_name() == "dtype") {
        LOG(INFO) << "Skip remove node " << it->kind().toQualString();
        continue;
      }
      LOG(INFO) << "Removed redundant node " << it->kind().toQualString();
      it->output()->replaceAllUsesWith(it->inputs()[0]);
      it.destroyCurrent();
    }
  }
}

bool TorchModule::Load(const std::string& module_path, InferMode mode) {
  try {
    module_ = ::torch::jit::load(module_path);
    module_.eval();
    module_.to(c10::kCPU);
  } catch (const c10::Error& e) {
    LOG(ERROR) << "error when load model: " << e.msg();
    return false;
  }
  mode_ = mode;
  graph_ = module_.get_method("forward").graph();
  PreProcessGraph();
  return true;
}

bool TorchModule::EvalAll(const std::vector<c10::IValue>& inputs) {
  // 创建所有 output 的 Tuple Node
  torch::jit::value_list values;
  for (auto input : graph_->inputs()) {
    values.push_back(input);
  }

  for (auto node : graph_->nodes()) {
    for (auto output : node->outputs()) {
      values.push_back(output);
    }
  }
  auto tuple_node = graph_->createTuple(values);
  graph_->insertNode(tuple_node);

  // 将输出替换为要评估的
  auto* output = graph_->outputs()[0];
  graph_->eraseOutput(0);
  graph_->registerOutput(tuple_node->output());

  const c10::IValue result = module_.forward(inputs);

  // 缓存结果
  auto results = result.toTuple()->elements();
  for (size_t i = 0; i < values.size(); ++i) {
    context_[values[i]] = results[i];
  }

  // 恢复输出节点
  graph_->eraseOutput(0);
  tuple_node->destroy();
  graph_->registerOutput(output);

  PostProcessGraph();
  return true;
}

void TorchModule::PreProcessGraph() {
#if FWD_TORCH_VERSION > 160
  torch::jit::Inline(*graph_);
#endif  // NEW_TORCH_API

  // 这里涉及到 jit graph 的优化
  //  合并线性层的操作，t + addmm, matmul + add, matmul -> linear
  //  这个操作可以将 t + addmm, matmul + add, matmul 等操作合并成 linear
  //  层， 但如果 addmm, matmul
  //  等涉及到的是两个非常量矩阵乘法这个操作不会进行转换
  torch::jit::FuseLinear(graph_);

#ifdef USE_TORCH_CUSTOM_OPS
  // 注意这个 fuse 操作会改变 graph，目前只确保 EvalAll
  // 对这些节点得输入输出维度正确一致。
  torch::pass::FuseLrn(graph_);

  torch::pass::FuseAdaLin(graph_);

  torch::pass::FuseTransformerEncoder(graph_);

#endif  // USE_TORCH_CUSTOM_OPS
  RemoveUnusedInputs();
}

void TorchModule::PostProcessGraph() {
  //  移除图中多余的节点，这个过程仍然是必要的是因为多余节点会干扰节点对于输入是否是常量的判断，
  //      例如： c10::prim::GetAttr -> c10::aten::device -> aten:to ->
  //      c10::aten::add 这个过程 to 使得 c10::aten::add 无法 根据输入节点是否是
  //      c10::prim::GetAttr/c10::prim::Constant 来判断输入是否是常量类型
  RemoveRedundantNodesOnBlock(graph_->block());

  // graph_->dump();

#if FWD_TORCH_VERSION > 160
  torch::pass::FoldFloorDivide(graph_);
#endif  // NEW_TORCH_API
}

void TorchModule::RemoveUnusedInputs() {
  valid_inputs_.reserve(graph_->inputs().size());
  for (auto input : graph_->inputs()) {
    if (input->debugNameBase() == "self") continue;
    valid_inputs_.push_back(input);
    input_map_[input->debugNameBase()] = input;
  }
}

const c10::IValue& TorchModule::Get(const JitValue* value) const {
  const c10::IValue& ivalue = context_.at(value);

  // 对于Tensor类型
  if (ivalue.isTensor() && !ivalue.toTensor().is_contiguous()) {
    LOG(WARNING) << "Tensor " << value->debugName()
                 << " is not contiguous. Use contiguous method before copy data.";
  }

  return ivalue;
}

void TorchModule::Dump() const { graph_->dump(); }

void TorchModule::SetMaxBatchSize(int size) { max_batch_size_ = size; }

InferMode TorchModule::GetMode() const { return mode_; }

int TorchModule::GetMaxBatchSize() const { return max_batch_size_; }

const std::set<int>& TorchModule::GetUnusedInputSet() const { return unused_input_indices_; }

void TorchModule::Clear() { context_.clear(); }

const std::vector<const JitValue*>& TorchModule::Inputs() const { return valid_inputs_; }

at::ArrayRef<JitValue*> TorchModule::Outputs() const { return graph_->outputs(); }

FWD_TORCH_NAMESPACE_END
