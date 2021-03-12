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

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

#include "trt_engine/trt_common/trt_profiler.h"

FWD_NAMESPACE_BEGIN

SimpleProfiler::SimpleProfiler(const char* name, const std::vector<SimpleProfiler>& srcProfilers)
    : mName(name) {
  for (const auto& srcProfiler : srcProfilers) {
    for (const auto& rec : srcProfiler.mProfile) {
      auto it = mProfile.find(rec.first);
      if (it == mProfile.end()) {
        mProfile.insert(rec);
      } else {
        it->second.time += rec.second.time;
        it->second.count += rec.second.count;
      }
    }
  }
}

void SimpleProfiler::reportLayerTime(const char* layerName, float ms) {
  mProfile[layerName].count++;
  mProfile[layerName].time += ms;
  if (std::find(mLayerNames.begin(), mLayerNames.end(), layerName) == mLayerNames.end()) {
    mLayerNames.push_back(layerName);
  }
}

void SimpleProfiler::exportJSONProfile(const std::string& fileName) const {
  std::ofstream os(fileName, std::ofstream::trunc);
  os << "[" << std::endl << "  { \"name\" : \"" << mName << "\" }" << std::endl;

  double totalTimeMs = 0.0;
  for (const auto& elem : mProfile) {
    totalTimeMs += elem.second.time;
  }

  for (const auto& name : mLayerNames) {
    auto elem = mProfile.at(name);
    // clang off
    os << ", {"
       << " \"name\" : \"" << name
       << "\""
          ", \"timeMs\" : "
       << elem.time << ", \"averageMs\" : " << elem.time / elem.count
       << ", \"percentage\" : " << elem.time * 100.0 / totalTimeMs << " }" << std::endl;
    // clang on
  }
  os << "]" << std::endl;
}

std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value) {
  out << "========== " << value.mName << " profile ==========" << std::endl;
  float totalTime = 0;
  std::string layerNameStr = "TensorRT layer name";
  int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
  for (const auto& elem : value.mProfile) {
    totalTime += elem.second.time;
    maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
  }

  auto old_settings = out.flags();
  auto old_precision = out.precision();
  // Output header
  {
    out << std::setw(maxLayerNameLength) << layerNameStr << " ";
    out << std::setw(12) << "Runtime, "
        << "%"
        << " ";
    out << std::setw(12) << "Invocations"
        << " ";
    out << std::setw(12) << "Runtime, ms" << std::endl;
  }
  for (size_t i = 0; i < value.mLayerNames.size(); i++) {
    const std::string layerName = value.mLayerNames[i];
    auto elem = value.mProfile.at(layerName);
    out << std::setw(maxLayerNameLength) << layerName << " ";
    out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime)
        << "%"
        << " ";
    out << std::setw(12) << elem.count << " ";
    out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << std::endl;
  }
  out.flags(old_settings);
  out.precision(old_precision);
  out << "========== " << value.mName << " total runtime = " << totalTime
      << " ms ==========" << std::endl;

  return out;
}

FWD_NAMESPACE_END
