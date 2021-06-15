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
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

FWD_NAMESPACE_BEGIN

using json = nlohmann::json;

constexpr const char* INFER_MODE = "infer_mode";
constexpr const char* OPT_BATCH_SIZE = "opt_batch_size";
constexpr const char* MAX_BATCH_SIZE = "max_batch_size";
constexpr const char* UNUSED_INPUT_INDICES = "unused_input_indices";
constexpr const char* OUTPUT_INDICES = "output_indices";
constexpr const char* TORCH_MODULE_PATH = "torch_module_path";

bool EngineMetaData::LoadMetaData(const std::string& meta_file) {
  json meta_json;
  std::ifstream meta(meta_file);
  if (!meta.is_open()) {
    LOG(ERROR) << "Error loading engine meta file: " << meta_file;
    return false;
  }
  meta >> meta_json;
  meta.close();

  try {
    mode_ = static_cast<InferMode>(meta_json[INFER_MODE].get<int>());
    opt_batch_size_ = meta_json[OPT_BATCH_SIZE].get<int>();
    max_batch_size_ = meta_json[MAX_BATCH_SIZE].get<int>();
    std::set<int> unused_input_indices(meta_json[UNUSED_INPUT_INDICES].begin(),
                                       meta_json[UNUSED_INPUT_INDICES].end());
    unused_input_indices_.swap(unused_input_indices);
    std::vector<int> output_pos(meta_json[OUTPUT_INDICES].begin(), meta_json[OUTPUT_INDICES].end());
    output_pos_.swap(output_pos);

    torch_module_path_ = meta_json[TORCH_MODULE_PATH].get<std::string>();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Parse meta file : " << meta_file
               << " to json object failed. Please check meta file's format";
    return false;
  }

  return true;
}

bool EngineMetaData::SaveMetaData(const std::string& meta_file) const {
  json meta_json;
  meta_json[INFER_MODE] = static_cast<int>(mode_);
  meta_json[OPT_BATCH_SIZE] = opt_batch_size_;
  meta_json[MAX_BATCH_SIZE] = max_batch_size_;
  meta_json[UNUSED_INPUT_INDICES] = unused_input_indices_;
  meta_json[OUTPUT_INDICES] = output_pos_;
  meta_json[TORCH_MODULE_PATH] = torch_module_path_;

  std::ofstream meta(meta_file);
  if (!meta.is_open()) {
    LOG(ERROR) << "create engine meta file " << meta_file << " failed";
    return false;
  }
  meta << std::setw(4) << meta_json << std::endl;
  meta.close();

  return true;
}

FWD_NAMESPACE_END
