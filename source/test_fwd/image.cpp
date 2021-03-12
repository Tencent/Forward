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

#include "test_fwd/image.h"

#include <string>
#include <vector>

Image::Image(const std::string& image_path) { image_ = cv::imread(image_path); }

Image::Image(const uchar* data, int height, int width, int channels) {
  const int type = SelectType(channels);
  if (type == -1) {
    std::cerr << "Image channels must be 1 or 3." << std::endl;
    return;
  }
  image_ = cv::Mat(height, width, type);
  memcpy(image_.data, data, sizeof(uchar) * height * width * channels);
  ConvertColor(false);
}

std::vector<int64_t> Image::Shape() const { return {image_.rows, image_.cols, image_.channels()}; }

uchar* Image::Data() const { return image_.data; }

Image& Image::Resize(int width, int height) {
  if (!IsOk()) {
    std::cerr << "Failed to resize: no image data" << std::endl;
  } else {
    cv::resize(image_, image_, {width, height});
  }
  return *this;
}

void Image::ConvertColor(bool from_bgr) {
  if (image_.channels() == 3) {
    if (from_bgr) {
      cv::cvtColor(image_, image_, cv::COLOR_BGR2RGB);
    } else {
      cv::cvtColor(image_, image_, cv::COLOR_RGB2BGR);
    }
  }
}

bool Image::Write(const std::string& image_path) const {
  if (!IsOk()) {
    std::cerr << "Failed to write image: no image data" << std::endl;
    return false;
  }

  cv::imwrite(image_path, image_);

  std::cout << "Image was written to " << image_path << std::endl;
  return true;
}

int Image::SelectType(int channels) {
  if (channels == 1) {
    return CV_8UC1;
  }
  if (channels == 3) {
    return CV_8UC3;
  }
  return -1;
}
