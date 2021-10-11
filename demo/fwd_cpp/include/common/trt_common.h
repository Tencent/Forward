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

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/fwd_common.h"

#undef max
#undef min

FWD_NAMESPACE_BEGIN

constexpr static int WARP_SIZE = 32;

namespace TrtCommon {
inline unsigned int GetElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Invalid DataType.");
      return 0;
  }
}

inline fwd::DataType FwdDataType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kINT32:
      return fwd::DataType::INT32;
    case nvinfer1::DataType::kFLOAT:
      return fwd::DataType::FLOAT;
    case nvinfer1::DataType::kHALF:
      return fwd::DataType::HALF;
    case nvinfer1::DataType::kINT8:
      return fwd::DataType::INT8;
    default:
      return fwd::DataType::INVALID;
  }
}

inline nvinfer1::DataType GetDataType(bool use_fp16, bool use_int8, bool is_calib_mode) {
  nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
  if (use_fp16) dtype = nvinfer1::DataType::kHALF;
  if (use_int8 && !is_calib_mode) dtype = nvinfer1::DataType::kINT8;
  return dtype;
}

inline bool CheckAndCopyFile(const std::string& dest_path, const std::string& src_path,
                             bool force_copy = false) {
  if (!force_copy) {
    // check if desc_file is existed
    std::ifstream file(dest_path);
    if (file.is_open()) {
      file.close();
      return true;
    }
    file.close();
  }

  std::ifstream src_file(src_path, std::ios::binary);
  if (!src_file.is_open()) {
    return false;
  }

  std::ofstream desc_file(dest_path, std::ios::binary);
  if (!desc_file.is_open()) {
    return false;
  }

  // copy
  desc_file << src_file.rdbuf();

  src_file.close();
  desc_file.close();

  return true;
}
//////////////////////////////////////////
//                                      //
//             TRT Network              //
//                                      //
//////////////////////////////////////////

static void SetTensorName(nvinfer1::ITensor* tensor, const std::string& prefix,
                          const std::string& name) {
  tensor->setName((prefix + name).c_str());
}

static void SetOutputName(nvinfer1::ILayer* layer, const std::string& prefix,
                          const std::string& name, int out_idx = 0) {
  SetTensorName(layer->getOutput(out_idx), prefix, name);
}

static bool SetOutputRange(nvinfer1::ILayer* layer, float max_val, int out_idx = 0) {
  return layer->getOutput(out_idx)->setDynamicRange(-max_val, max_val);
}

/**
 * \brief 重置最大工作空间大小，使其不超过当前可用显存的 4/5
 * \param size 最大工作空间大小
 * \return
 */
inline size_t ResetMaxWorkspaceSize(size_t size) {
  size_t free;
  size_t total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  // 限制一下需要的显存大小
  if (size > free * 4 / 5) {
    size = free * 4 / 5;
  }

  return size;
}

/**
 * \brief 根据网络输出的顺序，得到其对应的引擎绑定编号顺序
 *        即，将 binding 顺序调整为 markOutput 的顺序
 * \param engine 引擎
 * \param network 网络
 * \return 引擎绑定编号的输出顺序
 */
inline std::vector<int> GetOutputOrder(nvinfer1::ICudaEngine* engine,
                                       nvinfer1::INetworkDefinition* network) {
  std::vector<int> output_pos;
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    const auto output = network->getOutput(i);
    const int pos = engine->getBindingIndex(output->getName());
    output_pos.push_back(pos);
  }
  return output_pos;
}

//////////////////////////////////////////
//                                      //
//             Smart Pointer            //
//                                      //
//////////////////////////////////////////

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

}  // namespace TrtCommon

FWD_NAMESPACE_END
