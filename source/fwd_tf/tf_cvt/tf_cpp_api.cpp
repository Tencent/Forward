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

#include "fwd_tf/tf_cvt/tf_cpp_api.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fwd_tf/tf_cvt/tf_utils.h"

FWD_TF_NAMESPACE_BEGIN

Output Operation::Input(int index) const {
  auto input = TF_OperationInput(TF_Input{op_, index});
  return Output(graph_, input);
}

bool Operation::GetAttrBool(const char* attr_name) const {
  Status status;
  unsigned char value;
  TF_OperationGetAttrBool(op_, attr_name, &value, status);
  return value != 0;
}

int64_t Operation::GetAttrInt(const char* attr_name) const {
  Status status;
  int64_t value;

  TF_OperationGetAttrInt(op_, attr_name, &value, status);
  CHECK(status.Ok());

  return value;
}

std::vector<int64_t> Operation::GetAttrIntList(const char* attr_name) const {
  Status status;

  const TF_AttrMetadata metadata = TF_OperationGetAttrMetadata(op_, attr_name, status);
  CHECK(status.Ok() && metadata.is_list);

  std::vector<int64_t> list(metadata.list_size);

  TF_OperationGetAttrIntList(op_, attr_name, list.data(), metadata.list_size, status);
  CHECK(status.Ok());

  return list;
}

float Operation::GetAttrFloat(const char* attr_name) const {
  Status status;
  float value;

  TF_OperationGetAttrFloat(op_, attr_name, &value, status);
  CHECK(status.Ok());

  return value;
}

std::vector<float> Operation::GetAttrFloatList(const char* attr_name) const {
  Status status;

  const TF_AttrMetadata metadata = TF_OperationGetAttrMetadata(op_, attr_name, status);
  CHECK(status.Ok() && metadata.is_list);

  std::vector<float> results(metadata.list_size);
  TF_OperationGetAttrFloatList(op_, attr_name, results.data(), metadata.list_size, status);
  CHECK(status.Ok());

  return results;
}

std::string Operation::GetAttrString(const char* attr_name) const {
  Status status;

  const TF_AttrMetadata metadata = TF_OperationGetAttrMetadata(op_, attr_name, status);
  CHECK(status.Ok() && metadata.type == TF_ATTR_STRING);

  std::vector<char> buffer(metadata.total_size, 0);
  TF_OperationGetAttrString(op_, attr_name, buffer.data(), metadata.total_size, status);
  CHECK(status.Ok());

  return std::string{buffer.begin(), buffer.end()};
}

TF_DataType Operation::GetAttrType(const char* attr_name) const {
  TF_DataType data_type;
  Status status;

  TF_OperationGetAttrType(op_, attr_name, &data_type, status);
  CHECK(status.Ok());

  return data_type;
}

std::vector<TF_DataType> Operation::GetAttrTypeList(const char* attr_name) const {
  Status status;

  const TF_AttrMetadata metadata = TF_OperationGetAttrMetadata(op_, attr_name, status);
  CHECK_EQ(metadata.type, TF_ATTR_TYPE);
  CHECK(metadata.is_list);

  std::vector<TF_DataType> values(metadata.list_size);
  TF_OperationGetAttrTypeList(op_, attr_name, values.data(), metadata.list_size, status);
  CHECK(status.Ok());

  return values;
}

std::vector<int64_t> Operation::GetAttrShape(const char* attr_name) const {
  Status status;
  const TF_AttrMetadata metadata = TF_OperationGetAttrMetadata(op_, attr_name, status);
  CHECK(status.Ok() && metadata.type == TF_ATTR_SHAPE);

  std::vector<int64_t> buffer(metadata.total_size, 0);
  TF_OperationGetAttrShape(op_, attr_name, buffer.data(), metadata.total_size, status);
  CHECK(status.Ok());

  return buffer;
}

std::vector<std::vector<int64_t>> Operation::GetAttrShapeList(const char* attr_name) const {
  Status status;
  const TF_AttrMetadata metadata = TF_OperationGetAttrMetadata(op_, attr_name, status);
  CHECK(status.Ok() && metadata.type == TF_ATTR_SHAPE && metadata.is_list);

  std::vector<int64_t*> values(metadata.list_size);
  std::vector<int> values_ndims(metadata.list_size);
  std::vector<int64_t> storage(metadata.total_size);
  TF_OperationGetAttrShapeList(op_, attr_name, values.data(), values_ndims.data(),
                               metadata.list_size, storage.data(), metadata.total_size, status);
  CHECK(status.Ok());

  std::vector<std::vector<int64_t>> results;
  for (int i = 0; i < metadata.list_size; ++i) {
    std::vector<int64_t> shape(values_ndims[i]);
    shape.assign(values[i], values[i] + values_ndims[i]);
    results.push_back(std::move(shape));
  }

  return results;
}

Tensor Output::GetConstantTensor() const {
  TF_Output output{op_, index_};
  TF_Tensor* tensor = nullptr;
  Status status;
  TF_TryEvaluateConstant(graph_, output, &tensor, status);
  CHECK(status.Ok());
  return Tensor(tensor);
}

FWD_TF_NAMESPACE_END
