// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
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
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/fwd_common.h"

FWD_NAMESPACE_BEGIN

struct TrtNetworkDesc;

// Interface of ForwardEngine
class IForwardEngine {
 public:
  IForwardEngine() = default;

  virtual ~IForwardEngine() = default;

  // return true if engine_file is saved as engine_file.
  virtual bool Save(const std::string& engine_file) const = 0;

  // return true if engine_file is loaded as engine_file
  virtual bool Load(const std::string& engine_file) = 0;

  // return true if forwarding succeed. The vector of outputs store results of
  // forwarding. The data ptr in Tensors can be in host memory or device memory,
  // which is determined by DeviceType. The memory of the outputs is managed
  // internally, so the caller should NOT delete the data ptr in the Tensor of
  // outputs.
  virtual bool Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) = 0;

  // return true if forwarding succeed. The vector of outputs store results of
  // forwarding. The data ptr in Tensors can be in host memory or device memory,
  // which is determined by DeviceType. The memory of the outputs is managed
  // internally, so the caller should NOT delete the data ptr in the Tensor of
  // outputs.
  virtual bool ForwardWithName(const IOMappingVector& inputs, IOMappingVector& outputs) = 0;

  // return infer mode supported by this engine
  virtual InferMode GetMode() = 0;
};

// Interface of ForwardBuilder
class IForwardBuilder {
 public:
  IForwardBuilder() = default;

  virtual ~IForwardBuilder() = default;

  // return shared_ptr of IForwardEngine built by network_desc.
  // If building is failed, it returns nullptr.
  virtual std::shared_ptr<IForwardEngine> Build(const TrtNetworkDesc& network_desc) = 0;

  // InferMode should be set before Build. The default value is
  // InferMode::FLOAT.
  virtual void SetInferMode(InferMode mode) = 0;

  // opt_batch_size should be set before Build. If not set, the opt_batch_size
  // is set as the max_batch_size given by dummy inputs. The engine will try to
  // optimized itself for the input with the opt_batch_size.
  virtual void SetOptBatchSize(int size) = 0;
};

FWD_NAMESPACE_END
