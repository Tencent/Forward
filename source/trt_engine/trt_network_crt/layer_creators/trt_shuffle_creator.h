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

#include <vector>

#include "common/trt_utils.h"
#include "trt_engine/trt_network_crt/layer_creators/i_trt_layer_creator.h"

FWD_TRT_NAMESPACE_BEGIN

/**
 * \brief TRT 形状重整层创建器
 */
template <>
class TLayerCreator<TrtShuffleDesc> : public ILayerCreator {
 public:
  ITensorVector CreateLayer(nvinfer1::INetworkDefinition* network, const TrtLayerDesc* layer_desc,
                            const ITensorVector& input_tensors) override {
    const auto shuffle_desc = dynamic_cast<const TrtShuffleDesc*>(layer_desc);
    T_CHECK(shuffle_desc);

    auto& input = *input_tensors[0];

    if (shuffle_desc->channel_block_size > 0) {
      const bool is_NHWC = shuffle_desc->doFirstTrans && shuffle_desc->doSecondTrans;
      return is_NHWC ? CreateDepthToSpaceLayerNHWC(network, input, shuffle_desc)
                     : CreateDepthToSpaceLayerNCHW(network, input, shuffle_desc);
    }
    return CreateShuffleLayer(network, input, shuffle_desc);
  }

 private:
  ITensorVector CreateShuffleLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   const fwd::TrtShuffleDesc* const& shuffle_desc) {
    LOG(INFO) << "TrtShuffleDesc::CreateShuffleLayer";
    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(input);

    if (shuffle == nullptr) {
      LOG(ERROR) << "Create Network: Fail to create [shuffle] layer.";
      return {};
    }

    if (shuffle_desc->doFirstTrans) {
      shuffle->setFirstTranspose(shuffle_desc->firstTranspose);
    }
    if (shuffle_desc->doReshape) {
      // TODO(Ao Li): 支持 tf ExpandDims
      // for
      //   expand dims with more than 1 unknown dimension auto dim =
      //       shuffle_desc->reshapeDimensions;
      // auto input_dim = input.getDimensions();
      // for (int i = 1; i < dim.nbDims; i++) {
      //   if (dim.d[i] == -1) dim.d[i] = input_dim.d[i];
      // }
      // shuffle->setReshapeDimensions(dim);

      shuffle->setReshapeDimensions(shuffle_desc->reshapeDimensions);
    }
    if (shuffle_desc->doSecondTrans) {
      shuffle->setSecondTranspose(shuffle_desc->secondTranspose);
    }

    return {shuffle->getOutput(0)};
  }

  ITensorVector CreateDepthToSpaceLayerNHWC(nvinfer1::INetworkDefinition* network,
                                            nvinfer1::ITensor& input,
                                            const fwd::TrtShuffleDesc* const& shuffle_desc) {
    LOG(INFO) << "TrtShuffleDesc::CreateDepthToSpaceLayerNHWC";
    const int block_size = shuffle_desc->channel_block_size;
    const auto& final_dims = shuffle_desc->reshapeDimensions;

    nvinfer1::IShuffleLayer* first_shuffle = network->addShuffle(input);
    first_shuffle->setFirstTranspose(shuffle_desc->firstTranspose);  // NCHW -> NHWC

    // (..., H, W, C * scale_factor_1 * scale_factor_2)
    // => (..., H, W, scale_factor_1, scale_factor_2, C)
    // => (..., H, scale_factor_1, W, scale_factor_2, C)
    std::vector<int> first_dims(final_dims.d, final_dims.d + final_dims.nbDims - 3);
    first_dims.insert(first_dims.end(),
                      {final_dims.d[final_dims.nbDims - 3] / block_size,
                       final_dims.d[final_dims.nbDims - 2] / block_size, block_size, block_size,
                       final_dims.d[final_dims.nbDims - 1]});
    first_shuffle->setReshapeDimensions(TrtUtils::ToDims(first_dims));
    std::vector<int> permute;
    const int first_nbdims = first_dims.size();
    for (int i = 0; i < first_nbdims - 5; ++i) permute.push_back(i);  // dims before H
    permute.insert(permute.end(), {first_nbdims - 5, first_nbdims - 3, first_nbdims - 4,
                                   first_nbdims - 2, first_nbdims - 1});
    first_shuffle->setSecondTranspose(TrtUtils::ToPermutation(permute));

    // (..., H, upscale_factor_1, W, upscale_factor_2, C)
    // => (..., H * upscale_factor_1, W * upscale_factor_2, C)
    nvinfer1::IShuffleLayer* second_shuffle = network->addShuffle(*first_shuffle->getOutput(0));
    second_shuffle->setReshapeDimensions(final_dims);

    second_shuffle->setSecondTranspose(shuffle_desc->secondTranspose);  // NHWC -> NCHW
    return {second_shuffle->getOutput(0)};
  }

  ITensorVector CreateDepthToSpaceLayerNCHW(nvinfer1::INetworkDefinition* network,
                                            nvinfer1::ITensor& input,
                                            const fwd::TrtShuffleDesc* const& shuffle_desc) {
    LOG(INFO) << "TrtShuffleDesc::CreateDepthToSpaceLayer";
    const int block_size = shuffle_desc->channel_block_size;
    const auto& final_dims = shuffle_desc->reshapeDimensions;

    nvinfer1::IShuffleLayer* first_shuffle = network->addShuffle(input);
    // (..., C * scale_factor_1 * scale_factor_2, H, W)
    // => (..., C, scale_factor_1, scale_factor_2, H, W)
    // => (..., C, H, scale_factor_1, W, scale_factor_2)
    std::vector<int> first_dims(final_dims.d, final_dims.d + final_dims.nbDims - 2);
    first_dims.insert(first_dims.end(),
                      {block_size, block_size, final_dims.d[final_dims.nbDims - 2] / block_size,
                       final_dims.d[final_dims.nbDims - 1] / block_size});
    first_shuffle->setReshapeDimensions(TrtUtils::ToDims(first_dims));
    std::vector<int> permute;
    const int first_nbdims = first_dims.size();
    for (int i = 0; i < first_nbdims - 4; ++i) permute.push_back(i);  // dims before H
    permute.insert(permute.end(),
                   {first_nbdims - 2, first_nbdims - 4, first_nbdims - 1, first_nbdims - 3});
    first_shuffle->setSecondTranspose(TrtUtils::ToPermutation(permute));

    // (..., C, H, scale_factor_1, W, scale_factor_2)
    // => (..., C, H * scale_factor_1, W * scale_factor_2)
    nvinfer1::IShuffleLayer* second_shuffle = network->addShuffle(*first_shuffle->getOutput(0));
    second_shuffle->setReshapeDimensions(shuffle_desc->reshapeDimensions);

    return {second_shuffle->getOutput(0)};
  }
};

FWD_TRT_NAMESPACE_END
