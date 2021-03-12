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
#include <string>
#include <vector>

#include "common/common_macros.h"

FWD_NAMESPACE_BEGIN

/**
 * \brief Trt Engine Profiler
 */
class SimpleProfiler : public nvinfer1::IProfiler {
 public:
  struct Record {
    float time{0};
    int count{0};
  };

  /**
   * \brief 构造器
   * \param name
   * \param srcProfilers
   */
  SimpleProfiler(const char* name,
                 const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>());

  /**
   * \brief 报告 Engine 每一层的性能耗时
   * \param layerName
   * \param ms
   */
  void reportLayerTime(const char* layerName, float ms) override;

  /**
   * \brief 报告打印到 ostream
   * \param out
   * \param value
   * \return
   */
  friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value);

  /**
   * \brief 报告输出为 Json 文件
   * \param fileName
   */
  void exportJSONProfile(const std::string& fileName) const;

 private:
  std::string mName;

  std::vector<std::string> mLayerNames;

  std::map<std::string, Record> mProfile;
};

FWD_NAMESPACE_END
