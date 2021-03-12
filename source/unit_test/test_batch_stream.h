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

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "common/trt_batch_stream.h"
#include "common/trt_calibrator.h"

/**
 * \brief 正态分布随机数生成器
 */
class gen_norm {
 private:
  std::normal_distribution<float> distribution;
  std::default_random_engine rand_engine;

 public:
  explicit gen_norm(float _mean = 0.0, float _std = 1.0) : distribution(_mean, _std) {
    std::random_device rd;
    rand_engine = std::default_random_engine(rd());
  }
  float operator()() { return distribution(rand_engine); }
};

/**
 * \brief 随机输入float BatchStream
 */
class TestBatchStream : public fwd::IBatchStream {
 public:
  explicit TestBatchStream(std::vector<int64_t> size) : mSize(size) {
    for (auto& s : mSize) {
      mData.push_back(std::vector<float>(s));
    }
  }
  bool next() override { return mBatch < mBatchTotal; }

  // todo: this function is not thread safe
  std::vector<const void*> getBatch() override {
    std::vector<const void*> batch;
    if (mBatch < mBatchTotal) {
      ++mBatch;

      for (auto& d : mData) {
        std::generate(d.begin(), d.end(), gen_norm());
        batch.push_back(d.data());
      }

      return batch;
    }
    return {{}};
  }
  int getBatchSize() const override { return 1; }
  std::vector<int64_t> bytesPerBatch() const override {
    std::vector<int64_t> bytes;
    for (auto& val : mSize) bytes.push_back(val * sizeof(float));
    return bytes;
  }

 private:
  std::vector<int64_t> mSize;
  int mBatch{0};
  int mBatchTotal{500};
  std::vector<std::vector<float>> mData;
};

/**
 * \brief 随机输入Bert BatchStream
 */
class TestBertStream : public fwd::IBatchStream {
 public:
  TestBertStream(int batch_size, int seq_len, int emb_count)
      : mBatchSize(batch_size),
        mSeqLen(seq_len),
        mSize(batch_size * seq_len),
        mEmbCount(emb_count) {
    mData.resize(3, std::vector<int>(mSize));
  }

  bool next() override { return mBatch < mBatchTotal; }

  std::vector<const void*> getBatch() override {
    if (mBatch < mBatchTotal) {
      ++mBatch;
      std::vector<const void*> batch;

      for (int i = 0; i < mSize; i++) mData[0].push_back(rand() % mEmbCount);
      batch.push_back(mData[0].data());

      for (int i = 0; i < mBatchSize; i++) {
        int rand1 = rand() % (mSeqLen - 1) + 1;
        for (int j = 0; j < rand1; j++) mData[1][i * mSeqLen + j] = 1;
        for (int j = rand1; j < mSeqLen; j++) mData[1][i * mSeqLen + j] = 0;
      }
      batch.push_back(mData[1].data());

      for (int i = 0; i < mSize; i++) {
        mData[2][i] = rand() % 2;
      }
      batch.push_back(mData[2].data());

      return batch;
    }
    return {{}};
  }
  int getBatchSize() const override { return mBatchSize; }
  std::vector<int64_t> bytesPerBatch() const override {
    std::vector<int64_t> bytes(3, mSeqLen * sizeof(int));
    return bytes;
  }

 private:
  int64_t mSize;
  int mBatch{0};
  int mBatchTotal{500};
  int mEmbCount{0};
  int mBatchSize{0};
  int mSeqLen{0};
  std::vector<std::vector<int>> mData;  // input_ids, input_mask, segment_ids
};

/**
 * \brief 计算输入尺寸
 */

inline std::string getFilename(const std::string& path, char sep = '/') {
  std::string::size_type iPos = path.find_last_of(sep) + 1;
  std::string filename = path.substr(iPos, path.length() - iPos);
  return filename;
}
