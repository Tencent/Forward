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
#include <unordered_map>

#include "common/common_macros.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief 层级创建器管理器
 */
class LayerCreatorManager {
 public:
  /**
   * \brief 构造器
   */
  LayerCreatorManager();

  /**
   * \brief 析构器
   */
  virtual ~LayerCreatorManager() = default;

  /**
   * \brief 注册层级
   * \tparam T 层级描述
   */
  template <typename T>
  void RegisterCreator() {
    using LayerCreator = TLayerCreator<T>;
    name_to_creator_mapping_[T::NAME()] = std::make_shared<LayerCreator>();
  }

  /**
   * \brief 根据名称查找层级创建器
   * \param layer_name 层级名称
   * \return 层级创建器
   */
  ILayerCreator* FindCreator(const std::string& layer_name) const;

 private:
  /**
   * \brief 描述层级名称到创建器的映射
   */
  std::unordered_map<std::string, std::shared_ptr<ILayerCreator>> name_to_creator_mapping_;
};

FWD_TRT_NAMESPACE_END
