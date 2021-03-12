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

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <vector>

#include "unit_test/unit_test.h"

/**
 * \brief
 * 创建batch_size个长度为seq_length,且首部若干个值为1,其他值为0的向量，拼在一起
 */
template <typename T = int>
std::vector<T> CreateInputMask(int batch_size, int seq_length) {
  std::vector<T> res(batch_size * seq_length, 1);
  for (int i = 0; i < batch_size; i++) {
    int zero_st = rand() % seq_length + 1;
    for (int j = zero_st; j < seq_length; j++) {
      res[i * seq_length + j] = 0;
    }
  }
  return res;
}

class TestTfBert : public ::testing::Test {
 public:
  TestTfBert() {
    filename = std::string(tf_root_dir) + "../../data/bert/student.pb";
    output_names = {"strided_slice_1"};

    batch_size = 16;
    seq_length = 48;
    max_input_id = 18000;

    const auto input_id = (fwd::tf_::Utils::CreateRandomIntTensor<int32_t>(
        TF_INT32, {batch_size, seq_length}, max_input_id));  //  input_ids
    std::vector<int> mask = CreateInputMask(batch_size, seq_length);
    const auto input_mask = (fwd::tf_::Utils::CreateTensor(TF_INT32, {batch_size, seq_length},
                                                           mask.data()));  //  input_mask
    const auto segment_ids = (fwd::tf_::Utils::CreateRandomIntTensor<int32_t>(
        TF_INT32, {batch_size, seq_length}, 2));  //  segment_ids

    input_map_buffer["input_ids"] = input_id;
    input_map_buffer["input_mask"] = input_mask;
    input_map_buffer["segment_ids"] = segment_ids;

    for (auto& entry : input_map_buffer) {
      input_map[entry.first] = entry.second.get();
    }
  }

  ~TestTfBert() override = default;

 protected:
  void SetUp() override {}
  void TearDown() override {}

  int batch_size = 16;
  int seq_length = 48;
  int max_input_id = 18000;
  std::unordered_map<std::string, TF_Tensor*> input_map;
  std::unordered_map<std::string, std::shared_ptr<TF_Tensor>> input_map_buffer;
  std::string filename;
  std::vector<std::string> output_names;

 private:
  void TestBody() override {}
};

TEST_F(TestTfBert, Float) {
  TestTFInference(filename, "float32", input_map, output_names, 1e-3);
  // TestTfTime(filename, input_map, batch_size, "float32");
}

TEST_F(TestTfBert, Half) {
  TestTFInference(filename, "float16", input_map, output_names, 1e-2);
  // TestTfTime(filename,input_map, batch_size, "float32");
}

TEST_F(TestTfBert, Int8) {
  const auto batch_stream = std::make_shared<TestBertStream>(batch_size, seq_length, max_input_id);

  TestTfTime(filename, input_map, "int8_calib", 1, batch_stream);

  TestTFInference(filename, "int8", input_map, output_names, 1e-2, batch_stream);

  // TestTfTime(filename, input_map, "int8", 100, batch_stream);
}
