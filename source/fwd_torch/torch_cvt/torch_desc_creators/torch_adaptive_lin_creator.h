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

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_helper.h"

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief AdaptiveLin 层描述创建器
 */
template <>
class TLayerDescCreator<TrtAdaptiveLinDesc> : public ILayerDescCreator {
 public:
  bool Check(const JitNode* node, const TorchModule& module) override {
    if (node->kind() == c10::Symbol::fromQualString("fwd::adapt_lin")) {
      return true;
    }
    // 识别 adaptive layer instance normalization 模式
    return CheckAdaptLinPattern(node, module);
  }

  std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                       std::vector<const JitValue*>& input_values) override {
    LOG(INFO) << "TorchAdaptiveLinCreator::Create";
    double eps1;
    if (node->kind() == c10::Symbol::fromQualString("fwd::adapt_lin")) {
      const auto inputs = node->inputs();
      input_values.push_back(inputs[0]);
      input_values.push_back(inputs[1]);
      input_values.push_back(inputs[2]);

      if (!GetDouble(module.Get(inputs[3]), eps1)) return nullptr;
    } else {
      if (!CreateAdaptLinByPattern(node, module, input_values, eps1)) return nullptr;
    }

    auto layer_desc = std::make_shared<TrtAdaptiveLinDesc>();

    layer_desc->epsilon = static_cast<float>(eps1);
    layer_desc->max_batch_size = module.GetMaxBatchSize();

    return layer_desc;
  }

 private:
  /**
   * \brief 根据特定模式识别出来的 Normalization 层
   * \param node
   * \param module
   * \param input_values
   * \param eps1
   * \return
   */
  bool CreateAdaptLinByPattern(const JitNode* node, const TorchModule& module,
                               std::vector<const JitValue*>& input_values, double& eps1) const {
    const auto mul1_node = node->inputs()[0]->node();
    const auto mul2_node = node->inputs()[1]->node();

    const JitNode* div1_node;
    const JitNode* div2_node;
    const JitValue* input1;
    const JitValue* input2;

    // get rho and 1 - rho inputs
    GetInput(mul1_node, div1_node, input1);
    GetInput(mul2_node, div2_node, input2);

    // get epsilon
    const JitNode* add1_node;
    const JitNode* add2_node;
    double eps2;

    if (!GetEpsilon(module, div1_node, add1_node, eps1)) return false;
    if (!GetEpsilon(module, div2_node, add2_node, eps2)) return false;
    CHECK_EQ(eps1, eps2);

    const torch::jit::Node* var1_node;
    const torch::jit::Node* var2_node;
    c10::IntArrayRef reduce_dims1, reduce_dims2;
    bool unbiased1, unbiased2;
    GetReduceDims(module, add1_node, var1_node, reduce_dims1, unbiased1);
    GetReduceDims(module, add2_node, var2_node, reduce_dims2, unbiased2);

    // input0 : input
    // input1 : layer norm rho
    // input2 : instance norm rho
    const JitValue* const input = var1_node->inputs()[0];
    input_values.push_back(input);

    if (!SetRho(input_values, input1, input2, reduce_dims1, reduce_dims2)) return false;

    // PrintBias(module, reduce_dims1, unbiased1, unbiased2, input);
    return true;
  }

  /**
   * \brief 根据特定模式识别出来的 Normalization 层
   * \param node
   * \param module
   * \return
   */
  bool CheckAdaptLinPattern(const JitNode* node, const TorchModule& module) {
    if (!(node->kind() == c10::aten::add) || module.Get(node->inputs()[2]).toInt() != 1) {
      return false;
    }

    for (int i = 0; i < 2; ++i) {
      const JitNode* mul = node->inputs()[i]->node();
      if (!(mul->kind() == c10::aten::mul)) {
        return false;
      }
      const JitNode* div;
      if (!CheckDiv(mul, div)) return false;

      const JitNode* var;
      if (!CheckMeanAndVar(module, div, var)) return false;

      // check reduce dims
      const auto reduce_dims = ToIntVector(module.Get(var->inputs()[1]));
      if (reduce_dims != c10::IntArrayRef({1, 2, 3}) && reduce_dims != c10::IntArrayRef({2, 3})) {
        return false;
      }
    }
    return true;
  }

  bool GetDouble(const c10::IValue i_value, double& value) const {
    if (i_value.isTensor()) {
      value = i_value.toTensor().item().toDouble();
    } else if (i_value.isDouble()) {
      value = i_value.toDouble();
    } else {
      LOG(ERROR) << "Unsupported eps value " << i_value;
      return false;
    }
    return true;
  }

  void GetInput(const torch::jit::Node* const mul1_node, const JitNode*& div1_node,
                const JitValue*& input1) const {
    if (mul1_node->inputs()[0]->node()->kind() == c10::aten::div) {
      div1_node = mul1_node->inputs()[0]->node();
      input1 = mul1_node->inputs()[1];
    } else {
      div1_node = mul1_node->inputs()[1]->node();
      input1 = mul1_node->inputs()[0];
    }
  }

  bool GetEpsilon(const TorchModule& module, const JitNode* div_node, const JitNode*& add_node,
                  double& eps) const {
    add_node = div_node->inputs()[1]->node()->inputs()[0]->node();
    const torch::jit::IValue eps_value2 = module.Get(add_node->inputs()[1]);
    return GetDouble(eps_value2, eps);
  }

  void GetReduceDims(const TorchModule& module, const JitNode* add1_node,
                     const torch::jit::Node*& var1_node, c10::IntArrayRef& reduce_dims1,
                     bool& unbiased1) const {
    var1_node = add1_node->inputs()[0]->node();
    reduce_dims1 = ToIntVector(module.Get(var1_node->inputs()[1]));
    unbiased1 = module.Get(var1_node->inputs()[2]).toBool();
  }

  bool SetRho(std::vector<const JitValue*>& input_values, const JitValue* input1,
              const JitValue* input2, c10::IntArrayRef reduce_dims1,
              c10::IntArrayRef reduce_dims2) const {
    if (reduce_dims1 == c10::IntArrayRef({2, 3}) && reduce_dims2 == c10::IntArrayRef({1, 2, 3})) {
      input_values.push_back(input2);
      input_values.push_back(input1);
    } else if (reduce_dims1 == c10::IntArrayRef({1, 2, 3}) &&
               reduce_dims2 == c10::IntArrayRef({2, 3})) {
      input_values.push_back(input1);
      input_values.push_back(input2);
    } else {
      LOG(ERROR) << "Unsupported normalization pattern for reduce dims: " << reduce_dims1 << " and "
                 << reduce_dims2;
      return false;
    }
    return true;
  }

  void PrintBias(const TorchModule& module, c10::IntArrayRef reduce_dims1, bool unbiased1,
                 bool unbiased2, const JitValue* const input) const {
    // TODO(Ao Li): 这里需要乘上 sqrt((N - 1) / N) 来修正无偏方差
    if (unbiased1 || unbiased2) {
      const auto dummy = module.Get(input).toTensor();
      double n = 1.0;
      for (auto dim : reduce_dims1) {
        n *= dummy.size(dim);
      }
      if (n > 1 && (1.0 - std::sqrt((n - 1) / n)) > 1e-5) {
        LOG(WARNING) << "Normalization pattern may cause large deviation: "
                     << (1.0 - std::sqrt((n - 1) / n)) << " unbiased.";
      }
    }
  }

  bool CheckDiv(const JitNode* mul, const JitNode*& div) {
    if (mul->inputs()[0]->node()->kind() == c10::aten::div) {
      div = mul->inputs()[0]->node();
    } else if (mul->inputs()[1]->node()->kind() == c10::aten::div) {
      div = mul->inputs()[1]->node();
    } else {
      return false;
    }

    if (div->inputs().size() != 2) {
      return false;
    }
    return true;
  }

  bool CheckMeanAndVar(const TorchModule& module, const JitNode* div, const JitNode*& var) {
    const JitNode* sub = div->inputs()[0]->node();
    const JitNode* sqrt = div->inputs()[1]->node();
    if (!(sub->kind() == c10::aten::sub) || sub->inputs().size() <= 2 ||
        module.Get(sub->inputs()[2]).toInt() != 1  // check sub alpha
        || !(sqrt->kind() == c10::aten::sqrt) || sqrt->inputs().size() != 1) {
      return false;
    }

    const auto input = sub->inputs()[0]->node();
    const auto mean = sub->inputs()[1]->node();
    const auto add = sqrt->inputs()[0]->node();
    if (!(mean->kind() == c10::aten::mean) || !(add->kind() == c10::aten::add) ||
        !(add->inputs()[1]->node()->kind() == c10::prim::Constant ||
          add->inputs()[1]->node()->kind() == c10::prim::GetAttr) ||
        module.Get(add->inputs()[2]).toInt() != 1) {  // check add alpha
      return false;
    }

    var = add->inputs()[0]->node();
    if ((mean->inputs()[0]->node() != input) || !(var->kind() == c10::aten::var) ||
        (var->inputs()[0]->node() != input) ||
        ToIntVector(module.Get(mean->inputs()[1])) != ToIntVector(module.Get(var->inputs()[1]))) {
      return false;
    }
    return true;
  }
};

FWD_TORCH_NAMESPACE_END
