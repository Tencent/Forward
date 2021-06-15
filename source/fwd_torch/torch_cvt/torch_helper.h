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

#include <torch/script.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/fwd_weights.h"
#include "common/trt_utils.h"
#include "fwd_torch/fwd_torch_renaming.h"

#ifdef _MSC_VER
#undef max
#undef min
#endif

FWD_TORCH_NAMESPACE_BEGIN

/**
 * \brief 获取 at::Tensor 维度
 * \param tensor
 * \return
 */
static std::vector<int64_t> ShapeOf(const at::Tensor& tensor) {
  const int64_t n_dims = tensor.ndimension();
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < n_dims; ++i) {
    shape.push_back(tensor.size(i));
  }
  return shape;
}

/**
 * \brief 将 IValue 转换为 Tensors，并对 IValue 里面的 TensorList 和 Tuple
 * 类型进行拆包 \param value \param tensors
 */
static void ToTensors(const torch::jit::IValue& value, std::vector<at::Tensor>& tensors) {
  if (value.isTensor()) {
    tensors.push_back(value.toTensor());
  } else if (value.isTensorList()) {
    for (const auto& tensor : value.toTensorList()) {
      tensors.push_back(tensor);
    }
  } else if (value.isTuple()) {
    for (const torch::jit::IValue& element : value.toTuple()->elements()) {
      ToTensors(element, tensors);
    }
  } else {
    LOG(ERROR) << "Unsupported input type " << value.type()->str();
    CHECK(false);
  }
}

inline void ExtractAllInputs(const JitValue* input, std::vector<const JitValue*>& unpacked_inputs) {
  // 如果 input 是 list 或 tuple 类型，则展开
  auto kind = input->node()->kind();
  if (kind == c10::prim::ListConstruct || kind == c10::prim::TupleConstruct) {
    for (auto item : input->node()->inputs()) {
      ExtractAllInputs(item, unpacked_inputs);
    }
  } else {
    unpacked_inputs.push_back(input);
  }
}

/**
 * \brief 获取 Tensor 的维度
 * \param tensor
 * \return nvinfer1::Dims
 */
inline nvinfer1::Dims DimsOf(const at::Tensor& tensor) {
  CHECK_LE(tensor.ndimension(), nvinfer1::Dims::MAX_DIMS);
  const auto sizes = tensor.sizes();
  if (sizes.size() == 0 && tensor.numel() == 1) return {1, 1};

  nvinfer1::Dims dims{static_cast<int>(sizes.size()), {}};
  for (int64_t i = 0; i < sizes.size(); ++i) {
    dims.d[i] = static_cast<int>(sizes[i]);
  }
  return dims;
}

inline std::string ShapeStrOf(const at::Tensor& tensor) {
  return TrtUtils::ValueStrOf<int64_t>(ShapeOf(tensor));
}

inline std::string ShapeStrOf(const std::vector<nvinfer1::ITensor*>& tensors) {
  std::stringstream ss;
  ss << "[";
  if (!tensors.empty()) {
    ss << TrtUtils::ShapeStrOf(tensors[0]->getDimensions());
    for (size_t i = 1; i < tensors.size(); ++i) {
      ss << ", " << TrtUtils::ShapeStrOf(tensors[i]->getDimensions());
    }
  }
  ss << "]";
  return ss.str();
}

/**
 * \brief 将 Tensor 数据搬运到目标 Device 类型
 * \param inputs 输入
 * \param outputs 输出
 * \param device 目标 Device 类型
 */
inline void ToDevice(const c10::IValue& inputs, c10::IValue& outputs, c10::Device device) {
  if (inputs.isTensor()) {
    outputs = c10::IValue(inputs.toTensor().to(device));
  } else if (inputs.isTensorList()) {
    c10::List<at::Tensor> output;
    for (const auto& input : inputs.toTensorList()) {
      output.push_back(static_cast<at::Tensor>(input).to(device));
    }
    outputs = c10::IValue(output);
  } else if (inputs.isTuple()) {
    std::vector<c10::IValue> tuple = inputs.toTuple()->elements();
    std::vector<c10::IValue> output(tuple.size());
    for (size_t i = 0; i < tuple.size(); ++i) {
      ToDevice(tuple[i], output[i], device);
    }
    outputs = c10::ivalue::Tuple::create(output);
  } else if (inputs.isCapsule() || inputs.isBlob() || inputs.isObject()) {
    std::cerr << "object, capsule and blob type is Unsupported " << std::endl;
  } else {
    std::cerr << "Unsupported IValue type " << std::endl;
    assert(false);
  }
}

/**
 * \brief 将 Tensor 的 Half 数据转成 Float 类型
 * \param inputs 输入
 * \param outputs 输出
 */
inline void HalfToFloat(const c10::IValue& inputs, c10::IValue& outputs) {
  if (inputs.isTensor()) {
    auto t = inputs.toTensor();
    if (t.scalar_type() == c10::kHalf) {
      outputs = c10::IValue(t.to(c10::kFloat));
    } else {
      outputs = c10::IValue(t);
    }
  } else if (inputs.isTensorList()) {
    c10::List<at::Tensor> output;
    for (const auto& input : inputs.toTensorList()) {
      at::Tensor t = input;
      if (t.scalar_type() == c10::kHalf) {
        t = t.to(c10::kFloat);
      }
      output.push_back(t);
    }
    outputs = c10::IValue(output);
  } else if (inputs.isTuple()) {
    std::vector<c10::IValue> tuple = inputs.toTuple()->elements();
    std::vector<c10::IValue> output(tuple.size());
    for (size_t i = 0; i < tuple.size(); ++i) {
      HalfToFloat(tuple[i], output[i]);
    }
    outputs = c10::ivalue::Tuple::create(output);
  } else {
    std::cerr << "Unsupported IValue type ";
    assert(false);
  }
}

/**
 * \brief 预处理 Dummy inputs
 * \param inputs dummy inputs
 * \return
 */
inline bool RegularizeIValues(std::vector<c10::IValue>& inputs) {
  for (auto& input : inputs) {
    if (input.isTensor() || input.isTensorList() || input.isTuple()) {
      // 为了执行 EvalAll, dummy inputs 必须位于 cpu, 默认模型是 float 模型
      ToDevice(input, input, c10::kCPU);
      HalfToFloat(input, input);
    } else {
      LOG(ERROR) << "Unsupported IValue type, please use TensorType, "
                    "TensorListType, TupleType.";
      return false;
    }
  }
  return true;
}

/**
 * \brief 将 从 c10::IntArrayRef 类型转换为 nvinfer1::Dims
 * \param list 输入
 * \return nvinfer1::Dims
 */
inline nvinfer1::Dims ToDims(const c10::IntArrayRef& array) {
  CHECK_LE(array.size(), nvinfer1::Dims::MAX_DIMS);
  nvinfer1::Dims dims{static_cast<int>(array.size()),
                      {
                          0,
                      }};
  for (auto i = 0; i < array.size(); ++i) {
    dims.d[i] = array[i];
  }
  return dims;
}

/**
 * \brief 将 从 c10::List 类型转换为 nvinfer1::Dims
 * \param list 输入
 * \return nvinfer1::Dims
 */
inline nvinfer1::Dims ToDims(const c10::List<int64_t>& list) {
  CHECK_LE(list.size(), nvinfer1::Dims::MAX_DIMS);
  nvinfer1::Dims dims{static_cast<int>(list.size()),
                      {
                          0,
                      }};
  for (auto i = 0; i < list.size(); ++i) {
    dims.d[i] = list[i];
  }
  return dims;
}

/**
 * \brief 将 IValue 转换为 at::Tensor
 * \param inputs IValue输入
 * \return at::Tensor
 */
inline std::vector<at::Tensor> ToTensors(const std::vector<torch::jit::IValue>& inputs) {
  std::vector<at::Tensor> outputs;
  for (auto& input : inputs) {
    ToTensors(input, outputs);
  }
  return outputs;
}

inline std::string ShapeOrValueStrOf(const at::Tensor& tensor) {
  std::stringstream ss;
  if (tensor.numel() == 1) {
    ss << "value = ";
    if (tensor.item().isFloatingPoint())
      ss << tensor.item().toDouble();
    else if (tensor.item().isIntegral(true))
      ss << tensor.item().toLong();
    else if (tensor.item().isComplex())
      ss << tensor.item().toComplexDouble();
    else
      ss << "unknown scalar";
  } else {
    ss << "shape = " << ShapeStrOf(tensor);
  }
  return ss.str();
}

inline std::string StringOf(c10::ScalarType type) {
  switch (type) {
    case c10::kFloat:
      return "FLOAT";
    case c10::kHalf:
      return "HALF";
    case c10::kChar:
      return "CHAR";
    case c10::kLong:
      return "INT64";
    case c10::kInt:
      return "INT32";
    default:
      return "INVALID";
  }
}

template <typename T>
inline FwdWeights ToFwdWeights(const T& value) {
  float f_value = static_cast<float>(value);
  return FwdWeights(std::vector<float>{f_value});
}

/**
 * \brief 获取 Tensor 数据转换为 FwdWeights
 * \param tensor 输入
 * \return FwdWeights
 */
template <>
inline FwdWeights ToFwdWeights(const at::Tensor& tensor) {
  FwdWeights weights;

  at::Tensor real_tensor = tensor.contiguous();
  // TODO(Ao Li): Long 作为 Int 来处理
  if (real_tensor.scalar_type() == c10::kLong || real_tensor.scalar_type() == c10::kInt ||
      real_tensor.scalar_type() == c10::kDouble) {
    real_tensor = real_tensor.toType(c10::kFloat);
  }

  int factor = 1;

  switch (real_tensor.scalar_type()) {
    case c10::kHalf:
      weights.SetType(nvinfer1::DataType::kHALF);
      factor = 2;
      break;
    case c10::kChar:
      weights.SetType(nvinfer1::DataType::kINT8);
      break;
    case c10::kFloat:
      weights.SetType(nvinfer1::DataType::kFLOAT);
      factor = 4;
      break;
    default:
      LOG(ERROR) << "Unsupported tensor data type " << tensor.toString();
      CHECK(false);
  }

  weights.SetCount(real_tensor.numel());
  weights.SetDims(DimsOf(real_tensor));

  assert(weights.Count() > 0 && weights.Dims().nbDims > 0);

  const char* data_ptr = static_cast<const char*>(real_tensor.data_ptr());
  weights.SetData(data_ptr, real_tensor.numel() * factor);

  return weights;
}

/**
 * \brief 拆解 JitValue
 * \param input
 * \return
 */
inline std::vector<const JitValue*> UnpackJitValue(const JitValue* input) {
  std::vector<const JitValue*> unpakced_inputs;
  if (input->type()->kind() == c10::TypeKind::TensorType) {
    if (input->hasUses()) {
      unpakced_inputs.push_back(input);
    }
  } else {
    const ::torch::jit::use_list& uses = input->uses();
    for (auto use : uses) {
      JitNode* node = use.user;
      if (node->kind() == c10::prim::TupleUnpack || node->kind() == c10::prim::ListUnpack) {
        for (auto output : node->outputs()) {
          if (output->hasUses()) {
            unpakced_inputs.push_back(output);
          }
        }
      }
    }
  }
  return unpakced_inputs;
}

/**
 * \brief 拆解 IValue 中的 Tuple/List 类型变为 Tensor vector
 * \param inputs 输入
 * \return 拆解后的 输入
 */
inline std::vector<c10::IValue> UnpackIValues(const std::vector<torch::jit::IValue>& inputs) {
  std::vector<c10::IValue> unpacked_inputs;
  for (auto& input : inputs) {
    if (input.isTuple()) {
      const auto& tuple = input.toTuple()->elements();
      for (auto& entry : tuple) {
        unpacked_inputs.push_back(entry);
      }
    } else if (input.isTensorList()) {
      const auto& tensors = input.toTensorList();
      for (const auto& entry : tensors) {
        unpacked_inputs.emplace_back(entry);
      }
    } else if (input.isTensor()) {
      unpacked_inputs.push_back(input);
    } else {
      LOG(ERROR) << "Unsupported input!";
    }
  }
  return unpacked_inputs;
}

/**
 * \brief 从二进制文件中读取 tensor，tensor 类型必须以 float 存储
 * \param filename tensor 文件名
 * \param shape tensor 维度
 * \return 读取到的 tensor
 */
inline at::Tensor ReadFromBinary(const std::string& filename, const std::vector<int64_t>& shape) {
  std::ifstream fin(filename, std::ios::binary);
  at::Tensor tensor;
  if (fin.is_open()) {
    std::vector<float> data(
        std::accumulate(shape.begin(), shape.end(), 1ll, std::multiplies<int64_t>()));
    fin.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    tensor = ::torch::zeros(shape);
    memcpy(tensor.data_ptr(), data.data(), data.size() * sizeof(float));
    fin.close();
    return tensor;
  }
  return tensor;
}

inline void WriteToBinary(at::Tensor tensor, const std::string& filename) {
  tensor = tensor.contiguous();
  std::ofstream fout(filename, std::ios::binary);
  fout.write(static_cast<char*>(tensor.data_ptr()), tensor.nbytes());
  fout.close();
}

inline void WriteToText(at::Tensor tensor, const std::string& filename) {
  tensor = tensor.contiguous();
  std::ofstream fout(filename);
  fout << tensor << std::endl;
  fout.close();
}

FWD_TORCH_NAMESPACE_END
