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
#include "fwd_keras/keras_cvt/keras_cpp_api.h"

FWD_KERAS_NAMESPACE_BEGIN

/**
 * \brief 网络层描述创建器 接口
 */
class ILayerDescCreator {
 public:
  ILayerDescCreator() = default;

  virtual ~ILayerDescCreator() = default;

  /**
   * \brief 检查 层级 Json 是否符合 TRT 网络层描述创建要求
   * \param layer 层级 Json
   * \return 符合创建要求，返回 True；否则，返回 False
   */
  virtual bool Check(const Layer& layer) = 0;

  /**
   * \brief 创建 层级Json 对应的 网络层描述
   * \param layer 层级 Json
   * \param reader H5 模型加载器
   * \param input_names 层级输入
   * \return
   */
  virtual std::shared_ptr<TrtLayerDesc> Create(const Layer& layer, const H5ModelReader& reader,
                                               std::vector<std::string>& input_names) = 0;
};

/**
 * \brief 网络层描述创建器 模板类
 * \tparam T 网络层类型
 */
template <typename T>
class TLayerDescCreator;

FWD_KERAS_NAMESPACE_END
