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

// Macros definitions for Forward

#define FWD_NAMESPACE_BEGIN namespace fwd {
#define FWD_NAMESPACE_END }

#define FWD_TORCH_NAMESPACE_BEGIN \
  namespace fwd {                 \
  namespace torch_ {
#define FWD_TORCH_NAMESPACE_END \
  }                             \
  }

#define FWD_TF_NAMESPACE_BEGIN \
  namespace fwd {              \
  namespace tf_ {
#define FWD_TF_NAMESPACE_END \
  }                          \
  }

#define FWD_KERAS_NAMESPACE_BEGIN \
  namespace fwd {                 \
  namespace keras_ {
#define FWD_KERAS_NAMESPACE_END \
  }                             \
  }

#define FWD_TRT_NAMESPACE_BEGIN \
  namespace fwd {               \
  namespace trt_ {
#define FWD_TRT_NAMESPACE_END \
  }                           \
  }

#define CUDA_CHECK(status)                                                   \
  do {                                                                       \
    auto ret = (status);                                                     \
    if (ret != 0) {                                                          \
      std::cerr << "Cuda failure: " << cudaGetErrorString(ret) << std::endl; \
      abort();                                                               \
    }                                                                        \
  } while (0)

#define CUDNN_CHECK(call)                                                         \
  do {                                                                            \
    cudnnStatus_t status = call;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                         \
      std::cerr << "CUDNN failure: " << cudnnGetErrorString(status) << std::endl; \
      abort();                                                                    \
    }                                                                             \
  } while (0)

#define FWD_TORCH_VERSION (TORCH_VERSION_MAJOR * 100 + TORCH_VERSION_MINOR * 10 + TORCH_VERSION_PATCH)

