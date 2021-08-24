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
//          Zhaoyi LUO (luozy63@gmail.com)

#pragma once

#include <simple_profiler.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/i_forward_api.h"
#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_common/trt_meta_data.h"

FWD_NAMESPACE_BEGIN

struct TrtNetworkDesc;

/**
 * \brief 推理引擎构建类
 */
class TrtForwardBuilder : public IForwardBuilder {
 public:
  /**
   * \brief 构造函数
   */
  TrtForwardBuilder() = default;

  /**
   * \brief 析构函数
   */
  ~TrtForwardBuilder() = default;

  /**
   * \brief 根据 网络层描述 构建 TRT 推理引擎
   * \param network_desc 网络层描述
   * \return 推理引擎
   */
  std::shared_ptr<IForwardEngine> Build(const TrtNetworkDesc& network_desc) override;

  /**
   * \brief 构建 TRT 引擎
   * \param builder TensorRT 构建器
   * \param network TensorRT 网络定义
   * \return 成功，返回 True
   */
  nvinfer1::ICudaEngine* BuildEngine(nvinfer1::IBuilder* builder,
                                     nvinfer1::INetworkDefinition* network) const;

  /**
   * \brief 设置量化器
   * \param calibrator INT 量化器
   */
  void SetCalibrator(std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator);

  /**
   * \brief 设置最优批量值，引擎将针对此批量大小进行优化
   * \return size 最优批量值
   */
  void SetOptBatchSize(int size) override;

  /**
   * \brief 获取最优批量值
   * \return 最优批量值
   */
  int GetOptBatchSize() const;

  /**
   * \brief 设置最大批量值
   * \param size 最大批量值
   */
  void SetMaxBatchSize(int size);

  /**
   * \brief 获取最大批量值
   * \return 最大批量值
   */
  int GetMaxBatchSize() const;

  /**
   * \brief 设置推理模式
   * \param mode 推理模式
   */
  void SetInferMode(InferMode mode) override;

  /**
   * \brief 设置工作空间内存最大值
   * \param size 工作空间内存最大值
   */
  void SetMaxWorkspaceSize(size_t size);

  /**
   * \brief 获取工作空间内存最大值
   * \return 工作空间内存最大值
   */
  size_t GetMaxWorkspaceSize() const;

  /**
   * \brief 设置推理引擎元数据中的输出顺序
   * \param output_pos 推理引擎输出顺序
   */
  void SetOutputPositions(const std::vector<int>& output_pos);

  /**
   * \brief 获取 TRT 引擎的元数据
   * \return 推理引擎元数据
   */
  EngineMetaData GetEngineMetaData() const;

  /**
   * \brief 将基本网络信息网络输出到文件
   * \param network 待输出的网络
   * \param filename 输出到的文件名
   * \return 成功，返回 True
   */
  static bool DumpNetwork(const nvinfer1::INetworkDefinition* network,
                          const std::string& filename = "network.txt");

 protected:
  /**
   * \brief 设置动态 Batch 的配置
   * \param builder TensorRT 构建器
   * \param network TensorRT 网络定义
   * \param config TensorRT 构建器属性
   * \return 成功，返回 True
   */
  bool SetDynamicProfile(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network,
                         nvinfer1::IBuilderConfig* config) const;

  /**
   * \brief 设置 TRT 引擎的推理模式
   * \param builder TensorRT 构建器
   * \param config TensorRT 构建器属性
   */
  void SetBuilderConfig(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config) const;

  /**
   * \brief 构建推理引擎的最大 GPU 内存空间,最大可用内存空间大小
   */
  int64_t max_workspace_size_{1ll << 30};

  /**
   * \brief TRT 引擎的元数据
   */
  EngineMetaData meta_data_;

  /**
   * \brief 量化器指针
   */
  std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator_{nullptr};
};

FWD_NAMESPACE_END
