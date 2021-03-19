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
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN
enum class InferMode;
class IForwardEngine;
class IForwardBuilder;
struct Tensor;

/**
 * \brief Torch 模型推理加速 类
 */
class TorchEngine {
 public:
  /**
   * \brief 构造函数
   */
  explicit TorchEngine(std::shared_ptr<IForwardEngine> engine = nullptr);

  /**
   * \brief 析构函数
   */
  ~TorchEngine();

  /**
   * \brief 前向推理
   * \param input_map 网络输入
   * \return outputs 网络输出
   */
  std::vector<at::Tensor> ForwardWithName(
      const std::unordered_map<std::string, c10::IValue>& input_map) const;

  /**
   * \brief 前向推理
   * \param inputs 网络输入
   * \return outputs 网络输出
   */
  std::vector<at::Tensor> Forward(const std::vector<torch::jit::IValue>& inputs) const;

  /**
   * \brief 前向推理的单输入重载版本
   * @param input 输入数据
   * @return 输出
   */
  std::vector<at::Tensor> Forward(const torch::jit::IValue& input) const;

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
   * \brief 将 at::Tensor 转换为 Tensor
   * \param input_tensor torch tensor
   * \return fwd Tensor
   */
  Tensor ParseInput(at::Tensor& input_tensor) const;

  /**
   * \brief 将 网络输出结果数据 从 GPU 拷贝到 CPU 的 at::Tensor
   * \param buffers 网络输出的结果数据
   * \return at::Tensor 网络输出
   */
  std::vector<at::Tensor> CopyFromBuffers(const std::vector<Tensor>& buffers, bool use_cuda) const;

  /**
   * \brief 推理引擎
   */
  std::shared_ptr<IForwardEngine> engine_{nullptr};

  // 临时添加
  friend class TorchBuilder;
};

class TorchBuilder {
 public:
  TorchBuilder();

  ~TorchBuilder();

  /**
   * \brief 加载 Torch 模型，并利用 伪输入 构建推理引擎
   * \param module_path 模型文件路径
   * \param inputs 伪输入
   * \return 成功，返回True
   */
  std::shared_ptr<TorchEngine> Build(const std::string& module_path,
                                     const std::vector<torch::jit::IValue>& inputs);

  /**
   * \brief 加载 Torch 模型，并利用 伪输入 构建推理引擎
   * \param module_path 模型文件路径
   * \param input_map 伪输入
   * \return 成功，返回True
   */
  std::shared_ptr<TorchEngine> Build(const std::string& module_path,
                                     const std::unordered_map<std::string, c10::IValue>& input_map);

  /**
   * \brief 设置量化器
   */
  void SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator) const;

  /**
   * \brief 设置最优批量值，引擎将针对此批量大小进行优化
   * \return size 最优批量值
   */
  void SetOptBatchSize(int size) const;

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
