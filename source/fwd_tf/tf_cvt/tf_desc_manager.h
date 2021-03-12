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
#include <vector>

#include "fwd_tf/tf_cvt/tf_desc_creators/i_tf_layer_creator.h"

FWD_TF_NAMESPACE_BEGIN

/**
 * \brief Tf 描述创建器管理类
 */
class TfDescManager {
 public:
  /**
   * \brief 构造器
   */
  TfDescManager();

  /**
   * \brief 析构器
   */
  ~TfDescManager() = default;

  /**
   * \brief 遍历查找 TF_Operation 对应的层描述器
   * \param op TF_Operation 操作
   * \return Op 对应的层描述器；若未找到，则返回 nullptr。
   */
  ILayerDescCreator* FindDescCreator(const Operation& op);

 private:
  /**
   * \brief 注册 网络层描述 创建器
   * \tparam T 网络层描述类型
   */
  template <typename T>
  void RegisterCreator() {
    using TFLayerDescCreator = TLayerDescCreator<T>;
    layer_desc_creators_.push_back(std::make_shared<TFLayerDescCreator>());
  }

  /**
   * \brief 已注册的 网络层描述 创建器
   */
  std::vector<std::shared_ptr<ILayerDescCreator>> layer_desc_creators_;
};

FWD_TF_NAMESPACE_END
