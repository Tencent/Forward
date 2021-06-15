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

#include "fwd_torch/torch_cvt/torch_passes/fold_floor_divide.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>

// Note: from torch source, but just remove Constant If
namespace torch {

using JitNode = jit::Node;
using JitValue = jit::Value;

namespace pass {

namespace internal {

bool typeListEqual(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (*lhs[i] != *rhs[i]) {
      return false;
    }
  }
  return true;
}

bool tensorEqual(const at::Tensor& lhs, const at::Tensor& rhs) {
  return lhs.options().type_equal(rhs.options()) && lhs.equal(rhs);
}

template <typename attribute_type>  // int64_t, bool, double
bool attributesEqual(attribute_type a1, attribute_type a2) {
  return a1 == a2;
}

bool attributesEqual(const at::Tensor& a1, const at::Tensor& a2) { return tensorEqual(a1, a2); }

bool ivaluesEqual(const IValue& a1, const IValue& a2);

bool attributesEqual(const std::vector<at::Tensor>& lhs, const std::vector<at::Tensor>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), tensorEqual);
}

bool attributesEqual(at::ArrayRef<IValue> a1, at::ArrayRef<IValue> a2) {
  if (a1.size() != a2.size()) {
    return false;
  }
  for (size_t i = 0; i < a1.size(); ++i) {
    if (!ivaluesEqual(a1[i], a2[i])) {
      return false;
    }
  }
  return true;
}

bool attributesEqual(const IValue& a1, const IValue& a2) { return ivaluesEqual(a1, a2); }

// this is not a general-purpose comparison of IValues, it only covers the
// ivalues that are allowed as attributes, and it does not check type
// equivalence of containers.
bool ivaluesEqual(const IValue& a1, const IValue& a2) {
  if (a1.tagKind() != a2.tagKind()) {
    return false;
  }
  if (a1.isInt()) {
    return a1.toInt() == a2.toInt();
  }
  if (a1.isBool()) {
    return a1.toBool() == a2.toBool();
  }
  if (a1.isDouble()) {
    return a1.toDouble() == a2.toDouble();
  }
  if (a1.isTensor()) {
    return attributesEqual(a1.toTensor(), a2.toTensor());
  }
  if (a1.isNone()) {
    return true;
  }
  if (a1.isString()) {
    return a1.toStringRef() == a2.toStringRef();
  }
  if (a1.isList()) {
    return attributesEqual(a1.toListRef(), a2.toListRef());
  }
  if (a1.isTuple()) {
    at::ArrayRef<IValue> a1_elem = a1.toTuple()->elements();
    at::ArrayRef<IValue> a2_elem = a2.toTuple()->elements();
    return attributesEqual(a1_elem, a2_elem);
  }
  if (a1.isGenericDict()) {
    auto a1_dict = a1.toGenericDict();
    auto a2_dict = a2.toGenericDict();
    if (a1_dict.size() != a2_dict.size()) {
      return false;
    }

    auto it_a1 = a1_dict.begin();
    auto it_a2 = a2_dict.begin();

    while (it_a1 != a1_dict.end()) {
      const auto& e_a1 = *it_a1;
      const auto& e_a2 = *it_a2;

      if (!ivaluesEqual(e_a1.key(), e_a2.key()) || !ivaluesEqual(e_a1.value(), e_a2.value())) {
        return false;
      }
      it_a1++;
      it_a2++;
    }
    return true;
  }
  if (a1.isEnum()) {
    return a1.toEnumHolder() == a2.toEnumHolder();
  }
  TORCH_INTERNAL_ASSERT(false);
}

// Check whether two nodes have the same attributes in CSE.
// This function may be too conservative for general use.
// Do NOT support g/gs attributes.
bool attributesEqualCSE(const JitNode* lhs, const JitNode* rhs) {
  AT_ASSERT(lhs != nullptr);
  AT_ASSERT(rhs != nullptr);
  // One has attributes, the other does not.
  if (lhs->hasAttributes() != rhs->hasAttributes()) return false;
  // Neither has attributes.
  if (!lhs->hasAttributes() && !rhs->hasAttributes()) return true;

  auto lnames = lhs->attributeNames();
  auto rnames = rhs->attributeNames();
  std::sort(lnames.begin(), lnames.end());
  std::sort(rnames.begin(), rnames.end());
  if (lnames != rnames) return false;

  for (auto name : lnames) {
    if (lhs->kindOf(name) != rhs->kindOf(name)) return false;

#define COMPARE_ATTRIBUTEVALUE(selector)                                          \
  case torch::jit::AttributeKind::selector: {                                     \
    if (!attributesEqual(lhs->selector(name), rhs->selector(name))) return false; \
  } break;

    switch (lhs->kindOf(name)) {
      COMPARE_ATTRIBUTEVALUE(f)
      COMPARE_ATTRIBUTEVALUE(fs)
      COMPARE_ATTRIBUTEVALUE(i)
      COMPARE_ATTRIBUTEVALUE(is)
      COMPARE_ATTRIBUTEVALUE(s)
      COMPARE_ATTRIBUTEVALUE(ss)
      COMPARE_ATTRIBUTEVALUE(t)
      COMPARE_ATTRIBUTEVALUE(ts)
      COMPARE_ATTRIBUTEVALUE(ival)
      case torch::jit::AttributeKind::ty:
        if (*lhs->ty(name) != *rhs->ty(name)) {
          return false;
        }
        break;
      case torch::jit::AttributeKind::tys:
        if (!typeListEqual(lhs->tys(name), rhs->tys(name))) {
          return false;
        }
        break;
      case torch::jit::AttributeKind::g:
      case torch::jit::AttributeKind::gs:
        return false;
    }

#undef COMPARE_ATTRIBUTEVALUE
  }

  return true;
}

bool EqualNode(const JitNode* lhs, const JitNode* rhs) {
  if (lhs == nullptr && rhs == nullptr) return true;
  if (lhs == nullptr || rhs == nullptr) return false;

  if (!(lhs->kind() == rhs->kind())) return false;

  // Check whether the output types are the same.
  auto lhs_outputs = lhs->outputs();
  auto rhs_outputs = rhs->outputs();
  if (lhs_outputs.size() != rhs_outputs.size()) return false;
  for (size_t i = 0; i < lhs_outputs.size(); ++i) {
    if (*lhs_outputs[i]->type() != *rhs_outputs[i]->type()) return false;
  }

  // Check whether the inputs are the same.
  auto lhs_inputs = lhs->inputs();
  auto rhs_inputs = rhs->inputs();
  if (lhs_inputs.size() != rhs_inputs.size()) return false;
  if (!std::equal(lhs_inputs.begin(), lhs_inputs.end(), rhs_inputs.begin())) return false;

  if (!attributesEqualCSE(lhs, rhs)) return false;

  return true;
}
}  // namespace internal

struct ConstantPropagator {
  static void ConstantPropagation(torch::jit::Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      JitNode* n = *it;
      it++;  // advance iterator bc the current node may be destroyed
      ConstantPropagation(n);
    }
  }

 private:
  // An Op has runnable inputs if:
  // - All inputs are constants.
  // - It is an op that forwards tuples, and all inputs are constants
  // or tuples that we know the ivalue for. We can't use known tuple ivalues
  // for non-forwarding ops because that Tuple could contain an ivalue that is
  // not allowed as a constant, for instance, a Tensor with a gradient.
  static bool runnableInputs(JitNode* n) {
    if (std::all_of(n->inputs().begin(), n->inputs().end(),
                    [&](JitValue* v) { return v->node()->kind() == c10::prim::Constant; })) {
      return true;
    }
    return false;
  }

  static void ConstantPropagation(at::ArrayRef<torch::jit::Block*> blocks) {
    for (torch::jit::Block* block : blocks) {
      ConstantPropagation(block);
    }
  }

  static void inlineIfBody(torch::jit::Block* body) {
    JitNode* n = body->owningNode();
    for (auto it = body->nodes().begin(); it != body->nodes().end();) {
      JitNode* body_node = *it;
      // advance iterator because after body_node is moved its next pointer will
      // be to n
      it++;
      body_node->moveBefore(n);
    }
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
    }
    // NB: destroy the node here, because it might contain side effects, like
    // print
    n->destroy();
  }

  static void inlineIf(JitNode* n) {
    auto input_bool = jit::constant_as<bool>(n->input());
    AT_ASSERT(input_bool);
    const size_t block_index = *input_bool ? 0 : 1;
    ConstantPropagation(n->blocks().at(block_index));
    inlineIfBody(n->blocks().at(block_index));
  }

  static void replaceAndRemoveIfOutput(JitNode* n, size_t i, JitValue* replacement) {
    n->outputs().at(i)->replaceAllUsesWith(replacement);
    n->eraseOutput(i);
    n->blocks().at(0)->eraseOutput(i);
    n->blocks().at(1)->eraseOutput(i);
  }

  // remove extra outputs from the node
  static bool removeExtraIfOutputs(JitNode* n) {
    TORCH_CHECK(n->kind() == c10::prim::If, "Only supported for If nodes");
    auto true_block = n->blocks()[0];
    auto false_block = n->blocks()[1];
    auto graph = n->owningGraph();
    const auto initial_outputs = true_block->outputs().size();
    jit::WithInsertPoint guard(n);
    for (size_t i = 0; i < true_block->outputs().size();) {
      auto t_out = true_block->outputs().at(i);
      auto f_out = false_block->outputs().at(i);

      // neither block changes the output value
      if (true_block->outputs()[i] == false_block->outputs()[i]) {
        replaceAndRemoveIfOutput(n, i, true_block->outputs()[i]);
        continue;
      }

      // true block output is constant and constant matches false block output
      auto maybe_const = jit::toIValue(t_out);
      if (maybe_const && internal::EqualNode(t_out->node(), f_out->node())) {
        auto new_const = graph->insertConstant(*maybe_const);
        replaceAndRemoveIfOutput(n, i, new_const);
        continue;
      }

      i++;  // increment bc we didn't remove current index
    }
    // an output was removed
    return initial_outputs != true_block->outputs().size();
  }

  static void ConstantPropagation(JitNode* n) {
    const bool runnable_inputs = runnableInputs(n);
    if (n->kind() == c10::prim::If) {
      // inline node if we can, otherwise check for simplified outputs
      if (runnable_inputs) {
        inlineIf(n);
      } else {
        ConstantPropagation(n->blocks());
        removeExtraIfOutputs(n);
      }
    } else {
      ConstantPropagation(n->blocks());
    }
  }
};

}  // namespace pass
}  // namespace torch

// TODO(Ao Li): 这里强制将 torch1.7.0 中的 c10::prim::If 判定整数除法情况
// 改成了只处理浮点数除法，等同于 torch1.3.1 的处理，认为输入不会是整数
void torch::pass::FoldFloorDivide(std::shared_ptr<torch::jit::Graph>& graph) {
  JitValue* one{nullptr};
  for (JitNode* node : graph->nodes()) {
    if (node->kind() == c10::prim::Constant &&
        node->output()->type()->kind() == c10::TypeKind::BoolType &&
        node->i(node->attributeNames()[0]) == 1) {
      one = node->output();
    }
    if (node->kind() == c10::prim::If) {
      CHECK_NOTNULL(one);
      node->replaceInputWith(node->input(), one);
    }
  }

  ConstantPropagator::ConstantPropagation(graph->block());

  torch::jit::EliminateDeadCode(graph);
}
