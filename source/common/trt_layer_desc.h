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

#include <NvInfer.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/fwd_weights.h"

namespace torch {
namespace jit {
class Module;
}
}  // namespace torch

FWD_NAMESPACE_BEGIN

// NormalizationType for normalization plugins
enum class TrtNormalizationType {
  BATCH_NORMALIZATION = 0,

  INSTANCE_NORMALIZATION,

  LAYER_NORMALIZATION,

  LAYER_INSTANCE_NORMALIZATION,

  SKIP_LAYER_NORMALIZATION,
};

struct TrtLayerDesc;

// output structure for TrtLayerDesc
struct TrtLayerOutput {
  std::shared_ptr<TrtLayerDesc> layer_desc;
  int index{0};
};

// Interface of Layer descriptions for operators
struct TrtLayerDesc {
  virtual ~TrtLayerDesc() = default;

  virtual const std::string Name() const = 0;

  std::vector<TrtLayerOutput> inputs;
};

#define TRT_LAYER_DESC(LayerDesc)                  \
  static const char* NAME() { return #LayerDesc; } \
  const std::string Name() const override { return NAME(); }

// ConstantInput with constant Weights
struct ConstantInput {
  bool inUse{false};
  FwdWeights data;
  nvinfer1::Dims dim;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Layer descriptions of supported operators are declared below.
/// There are three groups of declarations:
///   1. Forward-customized: defined by Forward
///   2. TensorRT-related: mapping directly to TrtLayers;
///   3. Combination: defined by Forward; a combination of different kinds of operators
/// Declarations in the same group are organized in alphabetical order
///
/// 以下是各节点的描述类，按照项目组自定义、Trt定义、复合操作三类排序
/// 每类别中节点按照字母顺序排序
/// Trt定义中尚未实现的节点包括：Constant, Gather, Identity, RaggedSoftMax,
/// Scale, Shape, TopK
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////
//                                      //
//           Forward 自定义描述          //
//   Forward-Customized Descriptions    //
//                                      //
//////////////////////////////////////////

struct TrtOutputDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Output);
  std::string name;
};

struct TrtInputDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Input);

  std::string name;
  nvinfer1::DataType type;
  nvinfer1::Dims dimensions;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct TrtAdaptiveLinDesc : TrtLayerDesc {
  TRT_LAYER_DESC(TrtAdaptiveLinDesc);

  float epsilon;

  int max_batch_size{-1};
};

struct TrtAdaptivePoolDesc : TrtLayerDesc {
  TRT_LAYER_DESC(AdaptivePool);

  std::vector<int> output_size;
  nvinfer1::PoolingType pooling_type;
};

struct TrtBertDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Bert);

  int n_layers{0};
  int num_heads{0};
  int hidden_size{0};
  std::unordered_map<std::string, FwdWeights> weight_map;
  // standard bert model use gelu as default, but some models prefer to use relu.
  bool use_relu{false};
  bool use_fp16{false};
  bool use_int8{false};
  bool calib_mode{false};

  bool use_group_conv1d{false};
  // two fields below are kept for group conv1d
  nvinfer1::Dims intermediate_shape{0};
  int num_split{0};

  static constexpr const char* WQKV = "qkv_kernel";
  static constexpr const char* BQKV = "qkv_bias";
};

struct TrtCastDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Cast);

  nvinfer1::DataType otype;
};

struct TrtClampDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Clamp);

  bool has_min{true};
  bool has_max{true};
  float min{0.0f};
  float max{0.0f};
};

struct TrtConstantPadDesc : TrtLayerDesc {
  TRT_LAYER_DESC(ConstantPadNd);

  // [d0, d1], w0, w1, h0, h1
  std::vector<int> dims;
  float value;
};

struct TrtGeluDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Gelu);

  bool use_fp16{false};
  bool use_int8{false};
};

struct TrtGridSamplerDesc : TrtLayerDesc {
  TRT_LAYER_DESC(GridSampler);
  int interpolation_mode, padding_mode, align_corners;
};

struct TrtNormalizationDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Normalization);

  FwdWeights scales;
  FwdWeights bias;
  FwdWeights running_mean;
  FwdWeights running_var;
  float epsilon;
  bool affine;
  bool use_input_stats;
  TrtNormalizationType type;
  bool use_fp16;
  bool use_int8;
  bool use_calib_mode;

  // fields below are kept for skip layer norm
  int leading_dim;
  FwdWeights zeros;

  int max_batch_size{-1};
};

// Noop desc is used for operators that has no effect to network building and engine forwarding.
// Example:
//   1. c10::aten::feature_dropout, dropout
//   2. c10::prim::TupleConstruct
struct TrtNoopDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Noop);
  std::string debug_string;
};

struct TrtReflectionPadDesc : TrtLayerDesc {
  TRT_LAYER_DESC(ReflectionPad);
  std::vector<int> dims;
};

struct TrtEmbeddingBagDesc : TrtLayerDesc {
  TRT_LAYER_DESC(EmbeddingBag);
  FwdWeights data;
  int dim;
  int count;
  int offset;
  int op;
};

struct TrtIndexDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Index)

  int nbDims;
  int nbIndexDims;
  int nbIndex;
  std::vector<int> axis;
  std::vector<int> indices;
};

struct TrtNormDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Norm)

  float p, inv_p;  // p-norm and its reciprocal
  int64_t axes;
  int keepDim;
  bool div;
};

struct TrtRepeatDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Repeat)

  std::vector<int> repeats;
};

struct TrtSeparableConvDesc : TrtLayerDesc {
  TRT_LAYER_DESC(SeparableConv);

  int nbDepthOutputMaps;
  int nbPointOutputMaps;
  nvinfer1::Dims kernelSize;
  FwdWeights depthKernelWeights;
  FwdWeights pointKernelWeights;
  FwdWeights biasWeights;

  nvinfer1::Dims stride;
  nvinfer1::Dims dilation;

  nvinfer1::Dims prePadding;
  nvinfer1::Dims postPadding;
  nvinfer1::PaddingMode paddingMode{nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN};

  int nbGroups;
};

struct TrtSplitDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Split)

  int dim;
  std::vector<int> splitSize;
  std::vector<FwdWeights> chunk_sizes;

  bool dynamic_size{false};
};

struct TrtTorchModuleDesc : TrtLayerDesc {
  TRT_LAYER_DESC(TrtTorchModuleDesc);

  std::string module_path;
  std::vector<int> node_ids;
  std::vector<int> in_types;
  std::vector<nvinfer1::DataType> out_types;
  std::vector<nvinfer1::Dims> out_dims;
};

// This is added for TensorRT <= 7.0
struct TrtUpsampleBilinearDesc : TrtLayerDesc {
  TRT_LAYER_DESC(UpsampleBilinear);

  int output_h;
  int output_w;
  bool alignCorners;
  float scale_h{1.0f};
  float scale_w{1.0f};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////
//                                      //
//         TensorRT 层级相关描述         //
//    TensorRT-related Descriptions     //
//                                      //
//////////////////////////////////////////

struct TrtActivationDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Activation)

  nvinfer1::ActivationType activationType;
  // alpha is used in LeakyRelu, Elu, Selu, Softplus, Clip, HardSigmoid, ScaledTanh,
  // ThresholdedRelu
  float alpha{0};
  // beta is used in Selu, Softplus, Clip, HardSigmoid, ScaledTanh
  float beta{0};
};

struct TrtConcatenationDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Concatenation)

  int axis;
  bool is_stack{false};
};

struct TrtConstantDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Constant)

  FwdWeights weights;
  nvinfer1::Dims dimensions;
};

struct TrtConvolutionDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Convolution);

  int nbOutputMaps;
  nvinfer1::Dims kernelSize;
  FwdWeights kernelWeights;
  FwdWeights biasWeights;

  nvinfer1::Dims stride;
  nvinfer1::Dims dilation;

  nvinfer1::Dims prePadding;
  nvinfer1::Dims postPadding;
  nvinfer1::PaddingMode paddingMode{nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN};

  int nbGroups;

  // this is kept for Conv1D in Tensorflow
  int squeeze_dim{-1};
};

struct TrtDeconvolutionDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Deconvolution)

  int nbOutputMaps;
  nvinfer1::Dims kernelSize;
  FwdWeights kernelWeights;
  FwdWeights biasWeights;

  nvinfer1::Dims stride;
  nvinfer1::Dims prePadding;
  nvinfer1::Dims postPadding;

  // this param is not used.
  // nvinfer1::PaddingMode paddingMode;

  int nbGroups;
};

struct TrtElementWiseDesc : TrtLayerDesc {
  TRT_LAYER_DESC(ElementWise);

  ConstantInput inputs[2];
  nvinfer1::ElementWiseOperation operation;
};

struct TrtFullyConnectedDesc : TrtLayerDesc {
  TRT_LAYER_DESC(FullyConnected)

  int nbOutputChannels;
  FwdWeights kernelWeights;
  FwdWeights biasWeights;
};

struct TrtGatherDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Gather)

  ConstantInput inputs[2];
  int gatherAxis;
};

struct TrtIdentityDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Identity)

  ConstantInput input;
  bool copy{false};
};

struct TrtLRNDesc : TrtLayerDesc {
  TRT_LAYER_DESC(LRN)

  int windowSize;
  float alpha;
  float beta;
  float k;
};

struct TrtMatrixMultiplyDesc : TrtLayerDesc {
  TRT_LAYER_DESC(MatrixMultiply)

  ConstantInput inputs[2];
  nvinfer1::MatrixOperation op[2]{nvinfer1::MatrixOperation::kNONE,
                                  nvinfer1::MatrixOperation::kNONE};
};

struct TrtParametricReLUDesc : TrtLayerDesc {
  TRT_LAYER_DESC(ParametricReLU)

  FwdWeights slopWeights;
  nvinfer1::Dims slopDims;
};

struct TrtPoolingDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Pooling)

  nvinfer1::PoolingType poolingType;
  nvinfer1::Dims windowSize;
  nvinfer1::Dims stride;
  nvinfer1::Dims padding;
  nvinfer1::PaddingMode paddingMode{nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN};

  float blendFactor;
  bool averageCountExcludesPadding;
};

// TODO(yuanzexi): to be implemented
// struct TrtRaggedSoftMaxDesc : TrtLayerDesc

struct TrtReduceDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Reduce)

  nvinfer1::ReduceOperation operation;
  uint32_t reduceAxes{0u};
  bool keepDimensions{false};

  // variables below are kept for var/std
  bool isVarOp{false};
  bool isStdOp{false};
  bool unbiased;
  float bias;
};

struct TrtResizeDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Resize);

  nvinfer1::Dims outputDimensions;
  std::vector<float> scales;
  nvinfer1::ResizeMode resizeMode;
  bool alignCorners;
};

struct TrtRNNv2Desc : TrtLayerDesc {
  TRT_LAYER_DESC(RNNv2)

  int32_t layerCount;
  int32_t hiddenSize;
  int maxSeqLen;

  nvinfer1::RNNOperation operation;
  nvinfer1::RNNInputMode inputMode;
  nvinfer1::RNNDirection direction;

  std::vector<FwdWeights> weightsForGate;
  std::vector<FwdWeights> biasForGate;

  bool returnSequences{true};

  // Note: PyTorch only
  bool batchFirst{true};
  bool torchOrder{false};
};

struct TrtScaleDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Scale)

  nvinfer1::ScaleMode mode;
  FwdWeights shift;
  FwdWeights scale;
  FwdWeights power;

  // 这个参数是addScaleNd独有的
  // int channelAxis;
};

struct TrtSelectDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Select)

  ConstantInput trueInput, falseInput;
};

struct TrtShuffleDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Shuffle)

  nvinfer1::Permutation firstTranspose;
  nvinfer1::Dims reshapeDimensions;
  nvinfer1::Permutation secondTranspose;
  bool doFirstTrans, doReshape, doSecondTrans;
};

struct TrtSliceDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Slice)

  bool dynamic_start{false};
  bool dynamic_size{false};
  char squeeze_dim_flag{0};

  nvinfer1::Dims dummy_out_dims;
  nvinfer1::Dims start;
  nvinfer1::Dims size;
  nvinfer1::Dims stride;

  std::unordered_map<std::string, FwdWeights> weight_map;
};

struct TrtSoftmaxDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Softmax)

  uint32_t axes;
};

// TODO(yuanzexi): to be implemented
// struct TrtTopKDesc : TrtLayerDesc
// {
//     TRT_LAYER_DESC(TopK)

//     nvinfer1::TopKOperation operation;
//     nvinfer1::k k;
//     uint32_t reduceAxes;
// };

struct TrtUnaryDesc : TrtLayerDesc {
  TRT_LAYER_DESC(Unary)

  ConstantInput input;
  nvinfer1::UnaryOperation operation;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////
//                                      //
//          Forward 组合操作描述         //
//       Combination Descriptions       //
//                                      //
//////////////////////////////////////////

// This is the combination of MatMul and Add, which is mapping to c10::aten::addmm in Torch
struct TrtMatMulAddDesc : TrtLayerDesc {
  TRT_LAYER_DESC(MatMulAdd);

  float alpha{1};
  float beta{1};
};

FWD_NAMESPACE_END
