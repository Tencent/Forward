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
#include <string>
#include <vector>

#include "unit_test/unit_test_tf_helper.h"

TEST(TestTfRecommender, LastFM) {
  const std::string filename = std::string(tf_root_dir) + "lastfm.pb";
  const std::string engine_path = std::string(tf_root_dir) + "lastfm.pb.engine";

  const int batch_size = 100;
  const auto seq_item_id = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, 20}, 14598);
  const auto seq_len = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size}, 50);
  const auto user_gender = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size}, 3);
  const auto user_geo = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size}, 67);

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["inputgraph/seq_item_id"] = seq_item_id.get();
  input_map["inputgraph/seq_len"] = seq_len.get();
  input_map["inputgraph/user_gender"] = user_gender.get();
  input_map["inputgraph/user_geo"] = user_geo.get();
  const std::vector<std::string> output_names{"embedding_4/embedding_lookup/Identity_1"};

  TestTFInference(filename, "float32", input_map, output_names);
}

TEST(TestTfRecommender, DeepFM) {
  const std::string filename = std::string(tf_root_dir) + "DeepFM.pb";
  const std::string engine_path = std::string(tf_root_dir) + "DeepFM.pb.engine";

  const int batch_size = 1;
  std::vector<float> input_vec(1706, 0);
  input_vec[0] = 1;
  const auto input = fwd::tf_::CreateTensor(TF_FLOAT, {batch_size, 1706}, input_vec.data());
  const auto indices = fwd::tf_::CreateRandomIntTensor<int>(TF_INT32, {batch_size, 21}, 1706);

  std::unordered_map<std::string, TF_Tensor*> input_map;
  input_map["Placeholder"] = input.get();
  input_map["Placeholder_2"] = indices.get();
  const std::vector<std::string> output_names{"Softmax"};

  TestTFInference(filename, "float32", input_map, output_names);
}
