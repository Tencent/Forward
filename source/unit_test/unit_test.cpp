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

#include <gtest/gtest.h>

#ifdef ENABLE_TORCH

#include "unit_test/test_torch_nodes.h"
// #include "unit_test/test_torch_vision.h"
// #include "unit_test/test_torch_bert.h"

#endif  // ENABLE_TORCH

#ifdef ENABLE_TENSORFLOW

#include "unit_test/test_tf_nodes.h"
// #include "unit_test/test_tf_vision.h"
// #include "unit_test/test_tf_bert.h"

#endif  // ENABLE_TENSORFLOW

#ifdef ENABLE_KERAS

#include "unit_test/test_keras_nodes.h"
// #include "unit_test/test_keras_vision.h"

#endif  // ENABLE_KERAS

#ifdef ENABLE_ONNX

// #include "unit_test/test_onnx_models.h"
// #include "unit_test/test_onnx_dynamic.h"
// #include "unit_test/test_onnx_bert.h"

#endif  // ENABLE_ONNX

GTEST_API_ int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
