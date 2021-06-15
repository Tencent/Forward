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

/// Meta Data of Forward Engine. This meta data is required to Forward but not required to TensorRT.
/// The TRT engine built by TensorRT can be loaded by Forward with a handicraft meta data file.
class EngineMetaData {
 public:
  EngineMetaData() = default;

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

  void SetTorchModulePath(const std::string& path) { torch_module_path_ = path; }

  const std::string& TorchModulePath() const { return torch_module_path_; }

  bool LoadMetaData(const std::string& meta_file);

  bool SaveMetaData(const std::string& meta_file) const;

 private:
  InferMode mode_{InferMode::FLOAT};

  int max_batch_size_{-1};

  int opt_batch_size_{-1};

  std::string torch_module_path_;

  /// unused input indices in the given dummy inputs
  std::set<int> unused_input_indices_;

  /// output binding indices in the TRT engine.
  /// Because the order of MarkOutput in the Forward may not be the same as the order of outputs
  /// marked by PyTorch, the order of output indices should be saved in meta data.
  std::vector<int> output_pos_;
};

FWD_NAMESPACE_END
