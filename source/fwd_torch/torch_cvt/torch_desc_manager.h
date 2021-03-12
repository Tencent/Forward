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

#include <torch/script.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_macros.h"
#include "common/trt_network_desc.h"
#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"

FWD_TORCH_NAMESPACE_BEGIN
class TorchModule;

/**
 * \brief Torch 描述创建器管理类
 */
class TorchDescManager {
 public:
  /**
   * \brief 构造器
   */
  TorchDescManager();

  /**
   * \brief 析构器
   */
  ~TorchDescManager() = default;

  /**
   * \brief 遍历查找 JitNode 对应的层描述器
   * \param node JitNode
   * \param module 参数获取器
   * \return 对应的层描述器
   */
  ILayerDescCreator* FindDescCreator(const JitNode* node, const TorchModule& module);

 private:
  /**
   * \brief 注册 网络层描述 创建器
   * \tparam T 网络层描述类型
   */
  template <typename T>
  void RegisterCreator() {
    using TorchLayerDescCreator = TLayerDescCreator<T>;
    layer_desc_creators_.push_back(std::make_shared<TorchLayerDescCreator>());
  }

  /**
   * \brief 已注册的 网络层描述 创建器
   */
  std::vector<std::shared_ptr<ILayerDescCreator>> layer_desc_creators_;
};

FWD_TORCH_NAMESPACE_END
