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
#include <fstream>
#include <functional>
#include <numeric>
#include <string>

#include "common/trt_batch_stream.h"
#include "test_fwd/image.h"

std::vector<float> LoadImageToVector(const std::string& path, int height, int width) {
  Image image(path);
  if (!image.IsOk()) {
    std::cerr << "Failed to load image " << path;
    exit(-1);
  }

  image.Resize(width, height);
  image.ConvertColor(true);

  const std::vector<int64_t> dims = image.Shape();
  const auto options = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(false)
                           .is_variable(true);
  at::Tensor tensor = torch::empty(dims, options);
  memcpy(tensor.data_ptr(), image.Data(), tensor.numel() * sizeof(uint8_t));

  tensor = (tensor.to(c10::kFloat) - 127.5) / 127.5;  // normalize
  tensor = tensor.permute({2, 0, 1}).contiguous();    // HWC -> CHW
  tensor = tensor.unsqueeze(0);
  return std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
}

class ImgBatchStream : public fwd::IBatchStream {
 public:
  ImgBatchStream(const std::string& dataFilePath, const std::string& frameFilePath, int h, int w,
                 int batchSize = 1)
      : mBatchSize(batchSize), mDims{batchSize, 3, h, w} {
    getFileList(dataFilePath, mFilenames);
    getFileList(frameFilePath, mFrameNames);
    auto num_ele = std::accumulate(mDims.begin(), mDims.end(), 1, std::multiplies<int>());
    mSizes.push_back(num_ele);
    if (mFrameNames.size()) mSizes[0] *= 2;
    mMaxBatches = (mFilenames.size() + batchSize - 1) / batchSize;
  }

  bool next() override {
    if (mBatchCount >= mMaxBatches) {
      return false;
    }
    return true;
  }

  std::vector<std::vector<float>> getBatch() override {
    int start = mBatchCount * mBatchSize;
    int end = start + mBatchSize < mFilenames.size() ? start + mBatchSize : mFilenames.size();
    std::size_t image_size = mDims[1] * mDims[2] * mDims[3];
    std::vector<float> input_tensor;
    for (int i = start; i < end; ++i) {
      auto img = LoadImageToVector(mFilenames[i], mDims[2], mDims[3]);
      if (mFrameNames.size()) {
        auto frame = LoadImageToVector(mFrameNames[i], mDims[2], mDims[3]);
        img.insert(img.end(), frame.begin(), frame.end());
      }
      input_tensor.insert(input_tensor.end(), img.begin(), img.end());
    }
    ++mBatchCount;
    return {input_tensor};
  }

  int getBatchSize() const override { return mBatchSize; }

  std::vector<int64_t> size() const { return mSizes; }

 private:
  void getFileList(const std::string& filename, std::vector<std::string>& fileList) {
    std::ifstream fi(filename);
    if (!fi) {
      std::cout << "[ImgBatchStream] cannot open datafile " << filename << std::endl;
    }
    std::string line;
    while (getline(fi, line)) {
      fileList.push_back(strip(line));
    }
  }

  std::string strip(const std::string& str) {
    int start = 0;
    int end = str.size() - 1;
    while (start <= end) {
      if (str[start] == '\r' || str[start] == '\t' || str[start] == '\n' || str[start] == ' ') {
        ++start;
      } else if (str[end] == '\r' || str[end] == '\t' || str[end] == '\n' || str[start] == ' ') {
        --end;
      } else {
        break;
      }
    }
    if (start > end) {
      return "";
    }
    return str.substr(start, end - start + 1);
  }

  int mBatchSize;
  int mMaxBatches;
  int mBatchCount{0};
  std::vector<int> mDims;
  std::vector<int64_t> mSizes;
  std::vector<std::string> mFilenames;
  std::vector<std::string> mFrameNames;
};
