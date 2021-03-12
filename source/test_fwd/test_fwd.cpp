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

#include "common/trt_calibrator.h"
#include "fwd_torch/torch_engine/torch_engine.h"
#include "fwd_torch/torch_engine/torch_infer.h"
#include "test_fwd/image.h"
#include "test_fwd/img_batch_stream.h"

#ifdef _MSC_VER
const c10::DeviceType device = c10::kCPU;
#else
const c10::DeviceType device = c10::kCUDA;
#endif

at::Tensor LoadImageToTensor(const std::string& path, int height, int width) {
  std::cout << "Load input images and convert to torch tensor..." << std::endl;

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
                           .requires_grad(false);
  at::Tensor tensor = torch::empty(dims, options);
  memcpy(tensor.data_ptr(), image.Data(), tensor.numel() * sizeof(uint8_t));

  tensor = (tensor.to(c10::kFloat) - 127.5) / 127.5;  // normalize
  tensor = tensor.permute({2, 0, 1}).contiguous();    // HWC -> CHW
  return tensor.unsqueeze(0);                         // CHW -> NCHW
}

bool WriteTensorToImage(const at::Tensor& tensor, const std::string& path) {
  if (tensor.ndimension() != 3) {
    std::cerr << "Can not convert tensor ndimension != 3 to image" << std::endl;
    return false;
  }
  at::Tensor t = (tensor * 127.5 + 127.5).clamp(0, 255).to(torch::kUInt8);
  t = t.permute({1, 2, 0}).contiguous();  // CHW -> HWC
  const Image image(reinterpret_cast<uchar*>(t.data_ptr()), t.size(0), t.size(1), t.size(2));
  return image.Write(path);
}

void TestTorchInfer(const std::string& model_path, const std::string& user_image,
                    const std::string& frame_image, int height, int width, int output_index,
                    const std::string& output_path) {
  TorchInfer torch_infer;
  if (!torch_infer.LoadModel(model_path)) {
    return;
  }

  // load input
  at::Tensor input_tensor = LoadImageToTensor(user_image, height, width);

  // dual inputs
  if (!frame_image.empty()) {
    // concatenate
    const at::Tensor frame_tensor = LoadImageToTensor(frame_image, height, width);
    input_tensor = at::cat({input_tensor, frame_tensor}, 1);
  }

  // inference
  std::vector<at::Tensor> outputs =
      torch_infer.Forward({input_tensor.to(device)}, device == c10::kCUDA);

  // write output image
  WriteTensorToImage(outputs[output_index].squeeze(0).to(c10::kFloat).to(c10::kCPU), output_path);
}

void TestTrtApiInfer(const std::string& model_path, const std::string& engine_path,
                     const std::string& mode, const std::string& user_image,
                     const std::string& frame_image, int height, int width, int output_index,
                     const std::string& output_path, const std::string& calib_img_path = "",
                     const std::string& calib_frame_path = "") {
  // load input
  at::Tensor input_tensor = LoadImageToTensor(user_image, height, width);

  // dual inputs
  if (!frame_image.empty()) {
    // concatenate
    const at::Tensor frame_tensor = LoadImageToTensor(frame_image, height, width);
    input_tensor = at::cat({input_tensor, frame_tensor}, 1);
  }

  TorchBuilder torch_builder;
  if (mode == "int8") {
    std::shared_ptr<fwd::IBatchStream> ibs =
        std::make_shared<ImgBatchStream>(calib_img_path, calib_frame_path, height, width);
    std::shared_ptr<TrtInt8Calibrator> calib =
        std::make_shared<TrtInt8Calibrator>(ibs, engine_path + ".calib", "entropy");
    torch_builder.SetCalibrator(calib);
  }
  const auto torch_engine = torch_builder.Build(model_path, mode, {input_tensor});
  if (torch_engine == nullptr) {
    return;
  }

  if (!torch_engine->Save(engine_path)) {
    return;
  }

  TorchEngine engine;
  if (!engine.Load(engine_path)) {
    LOG(ERROR) << "Failed to load engine " << engine_path;
    return;
  }

  std::vector<at::Tensor> outputs = engine.Forward(input_tensor.to(device));
  if (outputs.empty()) {
    LOG(ERROR) << "Output is empty";
    return;
  }

  WriteTensorToImage(outputs[output_index].squeeze(0).to(c10::kFloat).to(c10::kCPU), output_path);
}

void TestResnet() {
#ifdef _MSC_VER
  const char* model_file = "../../../models/resnet18_jit.pth";
  const char* input_image = "../../data/image.png";
#else
  const char* model_file = "../../models/resnet18_jit.pth";
  const char* input_image = "image.png";
#endif
  const char* engine_path = "temp.engine";
  const char* img_calib = "images_for_calib.txt";

  TestTorchInfer(model_file, input_image, "", 224, 224, 0, "torch_output.jpg");
  TestTrtApiInfer(model_file, engine_path, "float32", input_image, "", 224, 224, 0,
                  "trt_fp32_output.jpg");
  TestTrtApiInfer(model_file, engine_path, "float16", input_image, "", 224, 224, 0,
                  "trt_fp16_output.jpg");
  TestTrtApiInfer(model_file, engine_path, "int8", input_image, "", 224, 224, 0,
                  "trt_int8_output.jpg", img_calib);
}

int main() {
  TestResnet();
  return 0;
}
