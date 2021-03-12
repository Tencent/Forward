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

#include <memory>
#include <string>
#include <vector>

#include "common/trt_layer_desc.h"
#include "fwd_torch/fwd_torch_renaming.h"
#include "fwd_torch/torch_version.h"

#ifdef _MSC_VER
// 关闭 Torch 太多的编译警告
#pragma warning(disable : 4005 4244 4251 4267 4275 4522)
#endif  // _MSC_VER

FWD_TORCH_NAMESPACE_BEGIN
class TorchModule;
#ifdef NEW_TORCH_API
inline std::vector<int64_t> ToIntVector(const c10::IValue& value) { return value.toIntVector(); }
#else
inline c10::IntArrayRef ToIntVector(const torch::jit::IValue& value) {
  return value.toIntListRef();
}
#endif

/**
 * \brief 网络层描述创建器 接口
 */
class ILayerDescCreator {
 public:
  ILayerDescCreator() = default;

  virtual ~ILayerDescCreator() = default;

  /**
   * \brief 检查 JitNode 是否符合 TRT 网络层描述创建要求
   * \param node JitNode
   * \param module IValue 访问器
   * \return 符合创建要求，返回 True；否则，返回 False
   */
  virtual bool Check(const JitNode* node, const TorchModule& module) = 0;

  /**
   * \brief 创建 JitNode 对应的 网络层描述
   * \param node JitNode
   * \param module IValue 访问器
   * \param node_inputs node 的输入
   * \return
   */
  virtual std::shared_ptr<TrtLayerDesc> Create(const JitNode* node, const TorchModule& module,
                                               std::vector<const JitValue*>& node_inputs) = 0;
};

/**
 * \brief 网络层描述创建器 模板类
 * \tparam T 网络层类型
 */
template <typename T>
class TLayerDescCreator;

FWD_TORCH_NAMESPACE_END
