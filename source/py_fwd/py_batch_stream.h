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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "common/trt_batch_stream.h"

class IPyBatchStream : public fwd::IBatchStream {
 public:
  std::vector<const void*> getBatch() override {
    std::vector<const void*> batch;
    auto numpy_batch = getNumpyBatch();
    for (auto& vec : numpy_batch) {
      batch.push_back(vec.data());
    }
    return batch;
  }

  virtual std::vector<py::array> getNumpyBatch() = 0;
};

class PyBatchStream : public IPyBatchStream {
 public:
  /* Inherit the constructors */
  using IPyBatchStream::IPyBatchStream;

  bool next() override {
    PYBIND11_OVERLOAD_PURE(bool,           /* Return type */
                           IPyBatchStream, /* Parent class */
                           next);          /* Name of function in C++ (must match Python name) */
  }

  std::vector<py::array> getNumpyBatch() {
    PYBIND11_OVERLOAD_PURE(std::vector<py::array>, /* Return type */
                           IPyBatchStream,         /* Parent class */
                           getNumpyBatch); /* Name of function in C++ (must match Python name) */
  }

  int getBatchSize() const override {
    PYBIND11_OVERLOAD_PURE(int,            /* Return type */
                           IPyBatchStream, /* Parent class */
                           getBatchSize);  /* Name of function in C++ (must match Python name) */
  }

  std::vector<int64_t> bytesPerBatch() const override {
    PYBIND11_OVERLOAD_PURE(std::vector<int64_t>, /* Return type */
                           IPyBatchStream,       /* Parent class */
                           bytesPerBatch); /* Name of function in C++ (must match Python name) */
  }
};
