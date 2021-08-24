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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN

class IForwardEngine;
class IForwardBuilder;
struct Tensor;
enum class InferMode;

/**
 * \brief ONNX 模型推理加速 类
 */
class OnnxEngine {
 public:
  /**
   * \brief 构造函数
   */
  explicit OnnxEngine(std::shared_ptr<IForwardEngine> engine = nullptr);

  /**
   * \brief 析构函数
   */
  ~OnnxEngine();

  /**
   * \brief 前向推理
   * \param inputs 网络输入
   * \param outputs 网络输出
   * \return 成功返回 True
   */
  bool Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const;

  /**
   * \brief 加载 TRT 推理引擎
   * \param engine_file 文件路径
   * \return 成功返回 True
   */
  bool Load(const std::string& engine_file) const;

  /**
   * \brief 保存 TRT 推理引擎
   * \param engine_file 文件路径
   * \return 成功返回 True
   */
  bool Save(const std::string& engine_file) const;

 private:
  /**
   * \brief 推理引擎
   */
  std::shared_ptr<IForwardEngine> engine_{nullptr};

  friend class OnnxBuilder;
};

class OnnxBuilder {
 public:
  OnnxBuilder();

  ~OnnxBuilder();

  /**
   * \brief 加载 ONNX 模型，并利用 批量大小 构建推理引擎
   * \param model_path 模型文件路径
   * \return 推理引擎
   */
  std::shared_ptr<OnnxEngine> Build(const std::string& model_path);

  /**
   * \brief 设置量化器
   */
  void SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator);

  /**
   * \brief 设置最优批量值，引擎将针对此批量大小进行优化
   * \param size 最优批量值
   */
  void SetOptBatchSize(int size);

  /**
   * \brief 设置最大批量值
   * \param size 最大批量值
   */
  void SetMaxBatchSize(int size);

  /**
   * \brief 设置推理模式
   * \param mode 推理模式
   * \return 成功返回 True
   */
  bool SetInferMode(const std::string& mode);

  /**
   * \brief 设置工作空间内存最大值
   * \param size 工作空间内存最大值
   */
  void SetMaxWorkspaceSize(size_t size);

 private:
  IForwardBuilder* builder_;

  InferMode mode_;

  /**
   * \brief 解析 ONNX 模型
   * \param model_path 模型文件路径
   * \param network 待写入的网络定义
   * \return 成功返回 True
   */
  bool ParseModelFromFile(const std::string& model_path,
                          nvinfer1::INetworkDefinition* network) const;

  /**
   * \brief 构建 TRT 引擎
   * \param builder TensorRT 构建器
   * \param network TensorRT 网络定义
   * \return 未初始化的 TRT 引擎
   */
  nvinfer1::ICudaEngine* BuildEngine(nvinfer1::IBuilder* builder,
                                     nvinfer1::INetworkDefinition* network);

  /**
   * \brief 获取最优批量值
   * \return 最优批量值
   */
  int GetOptBatchSize() const;

  /**
   * \brief 获取最大批量值
   * \return 最大批量值
   */
  int GetMaxBatchSize() const;

  /**
   * \brief 设置网络输入数据类型
   * \param network 解析 ONNX 模型后得到的网络定义
   * \return 成功返回 True
   */
  bool SetInputType(nvinfer1::INetworkDefinition* network) const;

  /**
   * \brief 设置推理引擎元数据中的输出顺序
   * \param output_pos 推理引擎输出顺序
   */
  void SetOutputPositions(const std::vector<int>& output_pos);
};

FWD_NAMESPACE_END
