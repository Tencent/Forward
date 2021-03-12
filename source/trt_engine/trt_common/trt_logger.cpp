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

#include "trt_engine/trt_common/trt_logger.h"

#include <string>
#include <vector>

#if ELPP_LOGGING_ENABLED
INITIALIZE_EASYLOGGINGPP
#endif  // ELPP_LOGGING_ENABLED

FWD_NAMESPACE_BEGIN

Logger::Logger(nvinfer1::ILogger::Severity level) : level_(level) {
#if ELPP_LOGGING_ENABLED
  int argc = 1;
  const char* argv[] = {"Forward"};
  START_EASYLOGGINGPP(argc, argv);

  el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Filename, "Forward_%datetime.log");
  el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format,
                                     "[%level] %datetime %fbase(%line): %msg");
  el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);
#endif  // ELPP_LOGGING_ENABLED
}

nvinfer1::ILogger& Logger::getTRTLogger() { return *this; }

// trt logger
void Logger::log(Severity severity, const char* msg) {
  if (severity > level_) {
    return;
  }

  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG(FATAL) << "[TRT] " << std::string(msg);
      break;
    case Severity::kERROR:
      LOG(ERROR) << "[TRT] " << std::string(msg);
      break;
    case Severity::kWARNING:
      LOG(WARNING) << "[TRT] " << std::string(msg);
      break;
    case Severity::kINFO:
      LOG(INFO) << "[TRT] " << std::string(msg);
      break;
    case Severity::kVERBOSE:
      LOG(INFO) << "[TRT] " << std::string(msg);
  }
}

Logger gLogger;

std::mutex mtx;

FWD_NAMESPACE_END
