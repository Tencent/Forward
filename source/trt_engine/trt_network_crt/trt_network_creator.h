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

#include <unordered_map>
#include <vector>

#include "common/trt_network_desc.h"
#include "trt_engine/trt_network_crt/trt_creator_manager.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TensorRT 网络创建器
 */
class TrtNetworkCreator {
 public:
  /**
   * \brief 构造器
   * \param network TensorRT 网络
   */
  explicit TrtNetworkCreator(nvinfer1::INetworkDefinition* network);

  /**
   * \brief 析构器
   */
  ~TrtNetworkCreator() = default;

  /**
   * \brief 创建网络
   * \param network_desc 网络描述
   * \return 成功返回 true，否则返回 false
   */
  bool Create(const TrtNetworkDesc& network_desc);

 private:
  /**
   * \brief 创建网络层
   * \param layer_desc 层描述
   * \return 层输出张量
   */
  ITensorVector CreateLayer(const TrtLayerDesc* layer_desc);

  /**
   * \brief 获取层级输入，若未创建，则先自动创建再获取
   * \param layer_inputs 层描述
   * \return 层级输入
   */
  ITensorVector GetLayerInputs(const std::vector<TrtLayerOutput>& layer_inputs);

  /**
   * \brief 网络结构定义
   */
  nvinfer1::INetworkDefinition* network_;

  /**
   * \brief 描述器到输出张量映射
   */
  std::unordered_map<const TrtLayerDesc*, ITensorVector> created_layers_;

  /**
   * \brief 层级创建器管理器
   */
  LayerCreatorManager creator_manager_;
};

FWD_TRT_NAMESPACE_END
