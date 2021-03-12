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

#include <simple_profiler.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/i_forward_api.h"
#include "trt_engine/trt_common/trt_common.h"
#include "trt_engine/trt_common/trt_meta_data.h"
#include "trt_engine/trt_common/trt_profiler.h"
#include "trt_engine/trt_engine/trt_buffer_manager.h"

FWD_NAMESPACE_BEGIN

struct TrtNetworkDesc;

/**
 * \brief 推理引擎类
 */
class TrtForwardEngine : public IForwardEngine {
 public:
  /**
   * \brief 构造函数
   */
  TrtForwardEngine(nvinfer1::ICudaEngine* engine = nullptr, const EngineMetaData& meta_data = {});

  /**
   * \brief 析构函数
   */
  ~TrtForwardEngine();

  /**
   * \brief 使用 device 指针参数的前向推理
   * \param inputs 输入 tensor 的指针 vector，可以指向 host memory, 也可以指向
   * device memory, 由 device_type 控制 \param outputs 作为返回值，接收输出
   * tensor, 输出 tensor 指向 device memory, 内存由内部管理，调用者不可销毁
   * \return 成功返回 true
   */
  bool Forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  /**
   * \brief 使用 device 指针参数的前向推理
   * \param inputs key=输入名, value=输入值 的指针 map，可以指向 host memory,
   * 也可以指向 device memory, 由 device_type 控制 \param outputs
   * 作为返回值，接收输出 tensor, 输出 tensor 指向 device memory,
   * 内存由内部管理，调用者不可销毁 \return 成功返回 true
   */
  bool ForwardWithName(const IOMappingVector& inputs, IOMappingVector& outputs) override;

  /**
   * \brief 保存 TRT 推理引擎
   * \param engine_file 文件路径
   * \return 成功返回 True
   */
  bool Save(const std::string& engine_file) const override;

  /**
   * \brief 加载 TRT 推理引擎
   * \param engine_file 文件路径
   * \return 成功返回 True
   */
  bool Load(const std::string& engine_file) override;

  /**
   * \brief 初始化 TRT 引擎
   * \return 成功，返回 True
   */
  bool InitEngine();

  /**
   * \brief 获取输入维度
   * \return
   */
  std::vector<std::vector<int>> GetInputDims() const;

  /**
   * \brief 获取输出维度
   * \return
   */
  std::vector<std::vector<int>> GetOutputDims() const;

  /**
   * \brief 获取输出数据类型
   * \param index 输出索引
   * \return 输出数据类型
   */
  DataType GetOutputType(int index) const;

  /**
   * \brief 获取推理模式
   * \return
   */
  InferMode GetMode() override;

 protected:
  /**
   * \brief 使用 device 指针参数的前向推理重载版本，不进行输入数量和维度的检查
   * \param inputs 输入 tensor 的指针 vector，可以指向 host memory, 也可以指向
   * device memory, 由 device_type 控制 \param outputs 作为返回值，接收输出
   * tensor, 输出 tensor 指向 device memory, 内存由内部管理，调用者不可销毁
   * \return 成功返回 true
   */
  bool Execute(const IOMappingVector& inputs, IOMappingVector& outputs);

  /**
   * \brief 从文件读取 Engine
   * \param engine_file 文件路径
   * \return 成功，返回 True
   */
  bool LoadEngine(const std::string& engine_file);

  /**
   * \brief 检查是否有 未被使用 的输入
   * \param inputs 输入
   * \return 成功返回 true
   */
  bool CheckInputNums(std::vector<Tensor>& inputs) const;

  /**
   * \brief 检查输入维度是否匹配
   * @param inputs 输入
   * @return 成功返回 true
   */
  bool CheckInputs(const IOMappingVector& inputs) const;

  /**
   * \brief 设置运行时的固定输入维度
   * \param batch_size 批量大小
   * \return
   */
  bool SetBindingDimensions(int batch_size);

  /**
   * \brief 输入的绑定编号
   */
  std::vector<int> input_binding_indices_;

  /**
   * \brief TRT 引擎的元数据
   */
  EngineMetaData meta_data_;

  /**
   * \brief 输入输出的 Device 内存管理
   */
  BufferManager buffer_manager_;

  /**
   * \brief TensorRT 推理引擎
   */
  TrtCommon::InferUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};

  /**
   * \brief TensorRT 推理执行上下文
   */
  TrtCommon::InferUniquePtr<nvinfer1::IExecutionContext> context_{nullptr};

  cudaStream_t stream_{nullptr};

#if TRT_INFER_ENABLE_PROFILING
  std::shared_ptr<utils::Profiler> profiler_{nullptr};

  std::unique_ptr<SimpleProfiler> trt_profiler_;
#endif  // TRT_INFER_ENABLE_PROFILING
};

FWD_NAMESPACE_END
