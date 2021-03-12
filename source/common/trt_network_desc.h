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

#include <memory>
#include <set>
#include <vector>

#include "common/trt_layer_desc.h"

FWD_NAMESPACE_BEGIN

// Network descriptions for building Trt engine
// Descriptions are organized as a network tree.
// Example:
//            TrtInputDesc_0            TrtInputDesc_1
//                  |                         |
//         TrtConvolutionDesc_0      TrtConvolutionDesc_1
//                  |                         |
//                  ---------------------------
//                               |
//                     TrtElementWiseDesc_0
//                               |
//                         TrtOutputDesc
//
struct TrtNetworkDesc {
  // OutputDesc will be handled as the start node of the network when building engine.
  std::vector<std::shared_ptr<TrtOutputDesc>> outputs;

  // InputDesc should be handled before handling the network
  std::vector<std::shared_ptr<TrtInputDesc>> inputs;

  // unused_input_indices are kept for the case that several inputs in the dummy input are not
  // used in the network.
  std::set<int> unused_input_indices;

  // max_batch_size for building engine
  int batch_size{-1};
};

FWD_NAMESPACE_END
