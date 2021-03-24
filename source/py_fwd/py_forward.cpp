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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef ENABLE_TORCH
#include "py_fwd/py_forward_torch.h"
#endif

#ifdef ENABLE_TENSORFLOW
#include "py_fwd/py_forward_tf.h"
#endif

#ifdef ENABLE_KERAS
#include "py_fwd/py_forward_keras.h"
#endif

#include "common/trt_calibrator.h"
#include "py_fwd/py_batch_stream.h"

namespace py = pybind11;

PYBIND11_MODULE(forward, m) {
  m.doc() = R"pbdoc(Forward for Python)pbdoc";

  m.attr("__version__") = "1.0.0";

  py::class_<IPyBatchStream, PyBatchStream, std::shared_ptr<IPyBatchStream>>(m, "IPyBatchStream")
      .def(py::init<>())
      .def("next", &IPyBatchStream::next)
      .def("getNumpyBatch", &IPyBatchStream::getNumpyBatch)
      .def("getBatchSize", &IPyBatchStream::getBatchSize)
      .def("bytesPerBatch", &IPyBatchStream::bytesPerBatch);

  // This calibrator is for compatibility with 2.0EA. It is deprecated and
  // should not be used. m.attr("LEGACY_CALIBRATION") = "legacy";
  m.attr("ENTROPY_CALIBRATION") = "entropy";
  m.attr("ENTROPY_CALIBRATION_2") = "entropy_2";
  m.attr("MINMAX_CALIBRATION") = "minmax";

  py::class_<nvinfer1::IInt8Calibrator, std::shared_ptr<nvinfer1::IInt8Calibrator>>(
      m, "IInt8Calibrator");

  py::class_<fwd::TrtInt8Calibrator, nvinfer1::IInt8Calibrator,
             std::shared_ptr<fwd::TrtInt8Calibrator>>(m, "TrtInt8Calibrator")
      .def(py::init<std::shared_ptr<IPyBatchStream>, const std::string&, const std::string&>())
      .def(py::init<const std::string&, const std::string&, int>())
      .def("set_scale_file", &fwd::TrtInt8Calibrator::setScaleFile);

#ifdef ENABLE_TORCH
  py::class_<fwd::TorchBuilder>(m, "TorchBuilder")
      .def(py::init<>())
      .def("build",
           py::overload_cast<fwd::TorchBuilder&, const std::string&,
                             const std::vector<torch::jit::IValue>&>(&TorchBuilderBuild),
           py::arg("module_path"), py::arg("dummy_inputs"))
      .def("build",
           py::overload_cast<fwd::TorchBuilder&, const std::string&, const torch::jit::IValue&>(
               &TorchBuilderBuild),
           py::arg("module_path"), py::arg("dummy_inputs"))
      .def("build_with_name",
           py::overload_cast<fwd::TorchBuilder&, const std::string&,
                             const std::unordered_map<std::string, c10::IValue>&>(
               &TorchBuilderBuildWithName),
           py::arg("module_path"), py::arg("dummy_input_dict"))
      .def("set_mode", &fwd::TorchBuilder::SetInferMode, py::arg("infer_mode"))
      .def("set_opt_batch_size", &fwd::TorchBuilder::SetOptBatchSize, py::arg("opt_batch_size"))
      .def("set_max_workspace_size", &fwd::TorchBuilder::SetMaxWorkspaceSize,
           py::arg("max_workspace_size"))
      .def("set_calibrator", &fwd::TorchBuilder::SetCalibrator, py::arg("calibrator"));

  py::class_<fwd::TorchEngine, std::shared_ptr<fwd::TorchEngine>>(m, "TorchEngine")
      .def(py::init<>())
      .def("forward",
           py::overload_cast<fwd::TorchEngine&, const std::vector<torch::jit::IValue>&>(
               &TorchEngineForward),
           py::arg("inputs"))
      .def("forward",
           py::overload_cast<fwd::TorchEngine&, const torch::jit::IValue&>(&TorchEngineForward),
           py::arg("input_tensor"))
      .def(
          "forward_with_name",
          py::overload_cast<fwd::TorchEngine&, const std::unordered_map<std::string, c10::IValue>&>(
              &TorchEngineForwardWithName),
          py::arg("input_dict"))
      .def("load", &fwd::TorchEngine::Load, py::arg("engine_filename"))
      .def("save", &fwd::TorchEngine::Save, py::arg("engine_filename"));
#endif  // ENABLE_TORCH

#ifdef ENABLE_TENSORFLOW
  py::class_<fwd::TfBuilder>(m, "TfBuilder")
      .def(py::init<>())
      .def("build", &TfBuilderBuild, py::arg("model_path"), py::arg("dummy_inputs"))
      .def("set_mode", &fwd::TfBuilder::SetInferMode, py::arg("infer_mode"))
      .def("set_opt_batch_size", &fwd::TfBuilder::SetOptBatchSize, py::arg("opt_batch_size"))
      .def("set_max_workspace_size", &fwd::TfBuilder::SetMaxWorkspaceSize,
           py::arg("max_workspace_size"))
      .def("set_calibrator", &fwd::TfBuilder::SetCalibrator, py::arg("calibrator"));

  py::class_<fwd::TfEngine, std::shared_ptr<fwd::TfEngine>>(m, "TfEngine")
      .def(py::init<>())
      .def("load", &fwd::TfEngine::Load, py::arg("engine_filename"))
      .def("save", &fwd::TfEngine::Save, py::arg("engine_filename"))
      .def("input_dims", &fwd::TfEngine::GetInputDims)
      .def("output_dims", &fwd::TfEngine::GetOutputDims)
      .def("forward", &TfEngineForwardWithName, py::arg("inputs"));

#endif  // ENABLE_TENSORFLOW

#ifdef ENABLE_KERAS
  py::class_<fwd::KerasBuilder>(m, "KerasBuilder")
      .def(py::init<>())
      .def("build", &fwd::KerasBuilder::Build, py::arg("model_path"), py::arg("batch_size") = 1)
      .def("set_mode", &fwd::KerasBuilder::SetInferMode, py::arg("infer_mode"))
      .def("set_opt_batch_size", &fwd::KerasBuilder::SetOptBatchSize, py::arg("opt_batch_size"))
      .def("set_max_workspace_size", &fwd::KerasBuilder::SetMaxWorkspaceSize,
           py::arg("max_workspace_size"))
      .def("set_calibrator", &fwd::KerasBuilder::SetCalibrator, py::arg("calibrator"));

  py::class_<fwd::KerasEngine, std::shared_ptr<fwd::KerasEngine>>(m, "KerasEngine")
      .def(py::init<>())
      .def("load", &fwd::KerasEngine::Load, py::arg("engine_filename"))
      .def("save", &fwd::KerasEngine::Save, py::arg("engine_filename"))
      .def("forward", &KerasEngineForward, py::arg("inputs"));
#endif  // ENABLE_KERAS
}
