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

#include <torch/script.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_macros.h"
#include "common/trt_network_desc.h"
#include "fwd_torch/torch_cvt/torch_desc_creators/i_torch_layer_creator.h"
#include "fwd_torch/torch_cvt/torch_desc_manager.h"
#include "fwd_torch/torch_cvt/torch_module.h"

// To solve the conflicts between internal LOG lib and Torch LOG lib
#ifdef LOG
#undef LOG
#endif

#define LOG(LEVEL) CLOG(LEVEL, ELPP_CURR_FILE_LOGGER_ID)

#include <easylogging++.h>

FWD_NAMESPACE_BEGIN
enum class InferMode;

FWD_NAMESPACE_END

FWD_TORCH_NAMESPACE_BEGIN

// Parse Torch JIT Module into NetworkDesc
class Parser {
 public:
  // constructor
  explicit Parser(InferMode mode);

  // Load Torch JIT Module and parse it into TrtNetwork descriptions with dummy inputs
  bool Parse(const std::string& module_path, const std::vector<torch::jit::IValue>& inputs);

  // Load Torch JIT Module and parse it into TrtNetwork descriptions with dummy input_map
  bool Parse(const std::string& module_path,
             const std::unordered_map<std::string, c10::IValue>& input_map);

  const TrtNetworkDesc& GetNetwork() const { return network_; }

  // Get InferMode
  InferMode GetMode() const { return mode_; }

  const std::set<int>& GetUnusedInputs() const { return network_.unused_input_indices; }

 private:
  // Utilize dummy_inputs to eval all outputs of intermediate JitValue, and then create
  // LayerDescs to these nodes.
  bool CreateDescs(const std::vector<c10::IValue>& inputs);

  // Create INPUT LayerDescs with dummy inputs
  bool CreateInputDescs(const std::vector<c10::IValue>& inputs);

  // Recursively parse JitValue into TrtLayerDesc, and regard these new created TrtLayerDesc as the
  // children(inputs) of parent.
  bool ParseValue(TrtLayerDesc* parent, const JitValue* value);

  // Set input type of TrtInputDesc
  bool SetInputType(std::shared_ptr<TrtInputDesc> input_desc,
                    const c10::ScalarType& input_type) const;

  // Fuse Torch Submodules, here is a simple example:
  //
  //     TrtInputDesc                    TrtInputDesc
  //          |                               |
  //   TrtTorchModuleDesc                     |
  //          |                 ==>  FusedTrtTorchModuleDesc
  //   TrtTorchModuleDesc                     |
  //          |                               |
  //     TrtOutputDesc                   TrtOutputDesc
  bool FuseTorchSubmodule(TrtLayerDesc* current);

  // Validate and set the batch size of network
  bool SetNetworkBatchSize();

  // Infer mode of the network
  InferMode mode_;

  // Mapping records of the created TrtLayerDesc and its corresponding JitValue
  std::unordered_map<const JitValue*, std::shared_ptr<TrtLayerDesc>> created_desc_map_;

  // The whole description of the network
  TrtNetworkDesc network_;

  // a TorchModule instance for loading Torch JIT Module and manipulating the JitValues
  TorchModule module_;

  // TrtLayerDesc Registry for JitValue
  TorchDescManager desc_manager_;
};

FWD_TORCH_NAMESPACE_END
