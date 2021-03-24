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

#include <gtest/gtest.h>

#ifdef ENABLE_TORCH

// #include "unit_test/test_torch_bert.h"
#include "unit_test/test_torch_nodes.h"
#include "unit_test/test_torch_nodes_fp16.h"
// #include "unit_test/test_torch_nodes_int8.h"
// #include "unit_test/test_torch_nodes_v170.h"
// #include "unit_test/test_torch_seg.h"
// #include "unit_test/test_torch_tuple_inputs.h"
// #include "unit_test/test_torch_vision.h"
// #include "unit_test/test_torch_vision_fp16.h"
// #include "unit_test/test_torch_vision_int8.h"

#endif  // ENABLE_TORCH

#ifdef ENABLE_TENSORFLOW

// #include "unit_test/test_tf_bert.h"
#include "unit_test/test_tf_nodes.h"
#include "unit_test/test_tf_nodes_fp16.h"
// #include "unit_test/test_tf_nodes_int8.h"
// #include "unit_test/test_tf_vision.h"

#endif  // ENABLE_TENSORFLOW

#ifdef ENABLE_KERAS

#include "unit_test/test_keras_nodes.h"
// #include "unit_test/test_keras_vision.h"

#endif  // ENABLE_KERAS

GTEST_API_ int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
