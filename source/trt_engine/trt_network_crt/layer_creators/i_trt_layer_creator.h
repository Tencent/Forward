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

#include <NvInfer.h>
#include <easylogging++.h>

#include <string>
#include <vector>

#include "common/trt_layer_desc.h"

FWD_TRT_NAMESPACE_BEGIN

using ITensorVector = std::vector<nvinfer1::ITensor*>;

/**
 * \brief TensorRT 层创建器
 */
class ILayerCreator {
 public:
  virtual ~ILayerCreator() = default;

  /**
   * \brief 根据层的描述，创建对应的层
   *
   * \param network 网络
   * \param layer_desc 层描述
   * \param input_tensors 层输入张量
   *
   * \return 层输出张量，失败返回 nullptr
   */
  virtual ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network,
                                    const TrtLayerDesc* layer_desc,
                                    const ITensorVector& input_tensors) = 0;
};

/**
 * \brief TensorRT 层创建器模板类
 *
 * \tparam T 层描述
 */
template <typename T>
class TLayerCreator;

FWD_TRT_NAMESPACE_END
