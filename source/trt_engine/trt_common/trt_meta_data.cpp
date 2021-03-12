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

#include "trt_engine/trt_common/trt_meta_data.h"

#include <easylogging++.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

FWD_NAMESPACE_BEGIN

bool EngineMetaData::LoadMetaData(const std::string& meta_file) {
  std::ifstream meta(meta_file);
  if (!meta.is_open()) {
    LOG(ERROR) << "Error loading engine meta file: " << meta_file;
    return false;
  }

  // infer mode
  int infer_mode;
  meta >> infer_mode;
  mode_ = static_cast<InferMode>(infer_mode);
  static const std::string MODE2STR[]{"FP32", "HALF", "INT8", "INT8_CALIB"};
  LOG(INFO) << "Set InferMode to " << MODE2STR[infer_mode];

  meta >> opt_batch_size_;
  LOG(INFO) << "Set opt_batch_size to " << opt_batch_size_;
  meta >> max_batch_size_;
  LOG(INFO) << "Set max_batch_size to " << max_batch_size_;

  // unused inputs
  int length, temp;
  meta >> length;
  LOG(INFO) << "Detect " << length << " unused inputs";
  unused_input_indices_.clear();
  for (int i = 0; i < length; ++i) {
    meta >> temp;
    unused_input_indices_.insert(temp);
  }

  // output positions
  meta >> length;
  output_pos_.clear();
  LOG(INFO) << "Detect " << length << " outputs";
  for (int i = 0; i < length; ++i) {
    meta >> temp;
    output_pos_.push_back(temp);
  }
  return true;
}

bool EngineMetaData::SaveMetaData(const std::string& meta_file) const {
  std::ofstream meta(meta_file);
  if (!meta.is_open()) {
    LOG(ERROR) << "create engine meta file " << meta_file << " failed";
    return false;
  }

  // save infer mode, int32
  meta << static_cast<int>(mode_) << "\n";

  // save opt_batch_size
  meta << opt_batch_size_ << "\n";

  // save max_batch_size
  meta << max_batch_size_ << "\n";

  // save unused inputs, length, u1, u2, ... total (length + 1) int
  meta << static_cast<int>(unused_input_indices_.size()) << " ";
  for (int elem : unused_input_indices_) {
    meta << elem << " ";
  }
  meta << "\n";

  // save output position, length, p1, p2, ... total (length + 1) int
  meta << static_cast<int>(output_pos_.size()) << " ";
  for (int elem : output_pos_) {
    meta << elem << " ";
  }
  meta << "\n";

  return true;
}

FWD_NAMESPACE_END
