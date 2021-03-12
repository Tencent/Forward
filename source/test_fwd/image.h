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

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class Image {
 public:
  Image() = default;

  /**
   * 从文件读取图像内容
   * \param image_path 图像文件路径
   */
  explicit Image(const std::string& image_path);

  /**
   * \brief 从数据构造图像
   * \param data 数据指针，指向 uint8 RGB 数据
   * \param height 图像高度
   * \param width 图像宽度
   * \param channels 图像通道数
   */
  Image(const uchar* data, int height, int width, int channels = 3);

  /**
   * \brief 获取图像的形状
   * \return 三维 vector, 表示 (height, width, channels)
   */
  std::vector<int64_t> Shape() const;

  /**
   * \brief 获取图像的数据指针
   * \return 图像的数据指针，nullptr 表示图像为空
   */
  uchar* Data() const;

  bool IsOk() const { return image_.data != nullptr; }

  /**
   * 改变图像大小
   * \param width 新的宽度
   * \param height 新的高度
   * \return this object
   */
  Image& Resize(int width, int height);

  /**
   * \brief 转换颜色模式，目前仅支持 RGB <-> BGR
   * \param from_bgr true 表示 BGR -> RGB, false 则相反；如果通道数不为
   * 3，则什么也不做
   */
  void ConvertColor(bool from_bgr);

  /**
   * 将图像写入到文件
   * \param image_path 图像写入路径
   * \return 成功返回 true
   */
  bool Write(const std::string& image_path) const;

 private:
  /**
   * \brief 根据通道数选择图像的类型
   * \param channels 通道数
   * \return 图像类型，如 CV_8UC1, CV_8UC3 等, 失败返回 -1
   */
  static int SelectType(int channels);

  cv::Mat image_;
};
