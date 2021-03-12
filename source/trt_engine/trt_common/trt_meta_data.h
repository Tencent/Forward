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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/common_macros.h"
#include "common/fwd_common.h"

FWD_NAMESPACE_BEGIN

/**
 * \brief TRT Engine 的元数据
 */
class EngineMetaData {
 public:
  /**
   * \brief 构造器
   */
  EngineMetaData() = default;

  /**
   * \brief 析构器
   */
  ~EngineMetaData() = default;

  InferMode Mode() const { return mode_; }

  void SetMode(InferMode mode) { mode_ = mode; }

  int MaxBatchSize() const { return max_batch_size_; }

  void SetMaxBatchSize(const int max_batch_size) { max_batch_size_ = max_batch_size; }

  int OptBatchSize() const { return opt_batch_size_; }

  void SetOptBatchSize(const int opt_batch_size) { opt_batch_size_ = opt_batch_size; }

  const std::set<int>& UnusedInputIndices() const { return unused_input_indices_; }

  void SetUnusedInputIndices(const std::set<int>& unused_input_indices) {
    unused_input_indices_ = unused_input_indices;
  }

  const std::vector<int>& OutputPositions() const { return output_pos_; }

  void SetOutputPositions(const std::vector<int>& output_pos) { output_pos_ = output_pos; }

  /**
   * \brief 从文件反序列化元数据
   * \param meta_file 文件路径
   * \return 是否加载成功
   */
  bool LoadMetaData(const std::string& meta_file);

  /**
   * \brief 保存元数据到文件
   * \param meta_file 文件路径
   * \return 是否保存成功
   */
  bool SaveMetaData(const std::string& meta_file) const;

 private:
  /**
   * \brief 推理模式
   */
  InferMode mode_{InferMode::FLOAT};

  /**
   * \brief 最大批量值
   */
  int max_batch_size_{-1};

  /**
   * \brief 最优批量值：TrtEngine 会针对该批量值进行性能优化
   */
  int opt_batch_size_{-1};

  /**
   * \brief 未被使用的 输入序号
   */
  std::set<int> unused_input_indices_;

  /**
   * 输出顺序，用于修正 TensorRT 输出位置是按照拓扑顺序进行 binding，
   * 导致输出与 Torch 顺序不一致的问题，SaveEngine时需要保存此信息
   */
  std::vector<int> output_pos_;
};

FWD_NAMESPACE_END
