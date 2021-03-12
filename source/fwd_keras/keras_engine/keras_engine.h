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
#include <NvInferRuntimeCommon.h>

#include <memory>
#include <string>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN

enum class InferMode;

class IForwardEngine;
class IForwardBuilder;

/**
 * \brief Keras 模型推理加速 类
 */
class KerasEngine {
 public:
  /**
   * \brief 构造函数
   */
  explicit KerasEngine(std::shared_ptr<IForwardEngine> engine = nullptr);

  /**
   * \brief 析构函数
   */
  ~KerasEngine();

  /**
   * \brief 加载 TRT 推理引擎
   * \param engine_path 文件路径
   * \return
   */
  bool Load(const std::string& engine_path);

  /**
   * \brief 保存 TRT  推理引擎
   * \param engine_path 文件路径
   * \return
   */
  bool Save(const std::string& engine_path) const;

  /**
   * \brief 推理
   * \param inputs
   * \param input_dims
   * \param outputs
   * \param output_dims
   * \param is_device
   * \return
   */
  bool Forward(const std::vector<void*>& inputs, const std::vector<std::vector<int>>& input_dims,
               std::vector<void*>& outputs, std::vector<std::vector<int>>& output_dims,
               bool is_device) const;

 private:
  std::shared_ptr<IForwardEngine> engine_{nullptr};

  friend class KerasBuilder;
};

class KerasBuilder {
 public:
  KerasBuilder();

  ~KerasBuilder();

  /**
   * \brief 加载 Keras 模型，并利用 批量大小 构建推理引擎
   * \param model_path 模型文件路径
   * \param batch_size 批量大小
   * \return 成功，返回 True
   */
  std::shared_ptr<KerasEngine> Build(const std::string& model_path, int batch_size = 1);

  /**
   * \brief 设置量化器
   */
  void SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator);

  /**
   * \brief 设置最大批量值
   * \return size 最大批量值
   */
  void SetOptBatchSize(int size);

  /**
   * \brief 设置推理模式
   * \param mode
   */
  bool SetInferMode(const std::string& mode);

  /**
   * \brief 设置工作空间内存最大值
   * \param size
   */
  void SetMaxWorkspaceSize(size_t size);

 private:
  IForwardBuilder* builder_;

  InferMode mode_;

  int max_batch_size_{-1};
};

FWD_NAMESPACE_END
