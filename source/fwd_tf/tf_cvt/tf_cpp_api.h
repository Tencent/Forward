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

#pragma once

#include <easylogging++.h>
#include <tensorflow/c/c_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common_macros.h"
#include "fwd_tf/tf_cvt/tf_graph.h"

FWD_NAMESPACE_BEGIN
enum class InferMode;

FWD_NAMESPACE_END

FWD_TF_NAMESPACE_BEGIN

class Status {
 public:
  Status() = default;

  ~Status() { TF_DeleteStatus(status_); }

  Status(const Status&) = delete;

  Status& operator=(const Status&) = delete;

  operator TF_Status*() const { return status_; }

  bool Ok() const { return Code() == TF_OK; }

  TF_Code Code() const { return TF_GetCode(status_); }

  const char* Message() const { return TF_Message(status_); }

 private:
  TF_Status* status_{TF_NewStatus()};
};

class Output;

class Operation {
 public:
  Operation() = default;

  explicit Operation(TF_Graph* graph, TF_Operation* op) : graph_(graph), op_(op) {
    CHECK((graph != nullptr) && (op_ != nullptr));
  }

  ~Operation() = default;

  std::string OpType() const { return TF_OperationOpType(op_); }

  std::string Name() const { return TF_OperationName(op_); }

  std::string Device() const { return TF_OperationDevice(op_); }

  int NumInputs() const { return TF_OperationNumInputs(op_); }

  TF_DataType InputType(int index) const { return TF_OperationInputType(TF_Input{op_, index}); }

  Output Input(int index) const;

  int NumOutputs() const { return TF_OperationNumOutputs(op_); }

  TF_DataType OutputType(int index) const { return TF_OperationOutputType(TF_Output{op_, index}); }

  int OutputNumConsumers(int index) const {
    return TF_OperationOutputNumConsumers(TF_Output{op_, index});
  }

  bool GetAttrBool(const char* attr_name) const;

  int64_t GetAttrInt(const char* attr_name) const;

  std::vector<int64_t> GetAttrIntList(const char* attr_name) const;

  float GetAttrFloat(const char* attr_name) const;

  std::vector<float> GetAttrFloatList(const char* attr_name) const;

  std::string GetAttrString(const char* attr_name) const;

  TF_DataType GetAttrType(const char* attr_name) const;

  std::vector<TF_DataType> GetAttrTypeList(const char* attr_name) const;

  std::vector<int64_t> GetAttrShape(const char* attr_name) const;

  std::vector<std::vector<int64_t>> GetAttrShapeList(const char* attr_name) const;

  TF_Graph* Graph() const { return graph_; }

  TF_Operation* Op() const { return op_; }

 protected:
  TF_Graph* graph_;
  TF_Operation* op_;
};

class Tensor {
 public:
  explicit Tensor(TF_Tensor* tensor) {
    if (tensor != nullptr) {
      tensor_ = std::shared_ptr<TF_Tensor>(tensor, TF_DeleteTensor);
    }
  }

  TF_DataType Type() const { return TF_TensorType(tensor_.get()); }

  int64_t ElementCount() const { return TF_TensorElementCount(tensor_.get()); }

  bool Valid() const { return tensor_ != nullptr; }

  template <typename T>
  T* Data() const {
    return static_cast<T*>(TF_TensorData(tensor_.get()));
  }

  size_t Size() const {
    return TF_TensorByteSize(tensor_.get()) / TF_DataTypeSize(TF_TensorType(tensor_.get()));
  }

  TF_Tensor* get() const { return tensor_.get(); }

  int AsInt() const {
    CHECK_EQ(this->Type(), TF_INT32);
    CHECK_EQ(this->Size(), 1);

    return *this->Data<int>();
  }

  std::vector<int> AsIntList() const {
    CHECK_EQ(this->Type(), TF_INT32);

    const int* data = this->Data<int>();
    return {data, data + this->Size()};
  }

 private:
  std::shared_ptr<TF_Tensor> tensor_;
};

class Graph;

class Output : public Operation {
 public:
  Output() = default;

  explicit Output(TF_Graph* graph, TF_Output& output)
      : Operation(graph, output.oper), index_(output.index) {}

  explicit Output(TF_Graph* graph, TF_Operation* op, int index)
      : Operation(graph, op), index_(index) {}

  int Index() const { return index_; }

  Tensor GetConstantTensor() const;

  int GetTensorNumDims() const {
    Status status;
    TF_Output output{op_, index_};
    auto dims = TF_GraphGetTensorNumDims(graph_, output, status);
    CHECK(status.Ok() && dims < 8);
    return dims;
  }

  bool GetTensorShape(int64_t* dims, int num_dims) const {
    Status status;
    TF_Output output{op_, index_};
    TF_GraphGetTensorShape(graph_, output, dims, num_dims, status);
    CHECK(status.Ok());
    return status.Ok();
  }

 private:
  int index_{-1};
};

FWD_TF_NAMESPACE_END
