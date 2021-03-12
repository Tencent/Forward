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
#include <tensorflow/c/tf_tensor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN

class IForwardEngine;
class IForwardBuilder;
struct Tensor;
enum class InferMode;

/**
 * \brief TensorFlow 模型推理加速 类
 */
class TfEngine {
 public:
  /**
   * \brief 构造函数
   */
  explicit TfEngine(std::shared_ptr<IForwardEngine> engine = nullptr);

  /**
   * \brief 析构函数
   */
  ~TfEngine();

  /**
   * \brief 前向推理
   * \param inputs 网络输入
   * \return outputs 网络输出 Tensor 集合, 指向 host memory
   */
  std::vector<std::shared_ptr<TF_Tensor>> Forward(const std::vector<TF_Tensor*>& inputs) const;

  /**
   * \brief 前向推理
   * \param inputs 网络输入 map: name -> TF_Tensor
   * \return outputs 网络输出 Tensor Map,指向 host memory
   */
  std::vector<std::pair<std::string, std::shared_ptr<TF_Tensor>>> ForwardWithName(
      const std::unordered_map<std::string, TF_Tensor*>& inputs) const;

  /**
   * \brief 加载 TRT 推理引擎
   * \param engine_file 文件路径
   * \return 成功返回 True
   */
  bool Load(const std::string& engine_file) const;

  /**
   * \brief 保存 TRT 推理引擎
   * \param engine_file 文件路径
   * \return
   */
  bool Save(const std::string& engine_file) const;

  /**
   * \brief 获取输入维度
   * \return
   */
  std::vector<std::vector<int>> GetInputDims() const;

  /**
   * \brief 获取输出维度
   * \return
   */
  std::vector<std::vector<int>> GetOutputDims() const;

 private:
  /**
   * \brief 将 网络输出结果数据 从 GPU 拷贝到 CPU 的 TF_Tensor
   * \param buffers 网络输出的结果数据
   * \return TF_Tensor 网络输出
   */
  std::vector<std::shared_ptr<TF_Tensor>> CopyFromBuffers(const std::vector<Tensor>& buffers) const;

  /**
   * \brief 将输入从 TF_Tensor 类型转换成 Tensor 类型
   * \param input 输入 TF_Tensor
   * \return new_input 新输入 Tensor
   */
  Tensor ParseInput(TF_Tensor* input) const;

  /**
   * \brief 推理引擎
   */
  std::shared_ptr<IForwardEngine> engine_{nullptr};

  friend class TfBuilder;
};

class TfBuilder {
 public:
  TfBuilder();

  ~TfBuilder();

  /**
   * \brief 加载 TensorFlow 模型，并利用 批量大小 构建推理引擎
   * \param model_path 模型文件路径
   * \param dummy_input_map 伪输入映射 name_string -> Tensor*
   * \return 成功，返回 True
   */

  std::shared_ptr<TfEngine> Build(
      const std::string& model_path,
      const std::unordered_map<std::string, TF_Tensor*>& dummy_input_map) const;

  /**
   * \brief 设置量化器
   */
  void SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator);

  /**
   * \brief 设置最优批量值，引擎将针对此批量大小进行优化
   * \return size 最优批量值
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
};

FWD_NAMESPACE_END
