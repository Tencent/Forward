# Models & Operators

----

- [Models](#models)
  - [CV](#cv)
  - [NLP](#nlp)
  - [Recommender](#recommender)
- [Operators](#operators)
  - [PyTorch](#pytorch)
  - [TensorFlow](#tensorflow)

----

## Models

### CV

| Model Name| PyTorch |TensorFlow|Keras|
| -- | -- |--|--|
| AlexNet | Yes |Yes|Yes|
| DenseNet121 | Yes |Yes|Yes|
| EfficientNet | Yes |Yes|Yes|
| GAN | Yes |Yes|Yes|
| GoogLeNet | Yes |Yes|Yes|
| Inception_v3 | Yes |Yes|Yes|
| InceptionResNetV2 | Yes |Yes|Yes|
| MNASNet0_75 | Yes |Yes|Yes|
| Mobilenet_v2 | Yes |Yes|Yes|
| ResNet50 | Yes |Yes|Yes|
| ShuffleNet_v2_x1_5 | Yes |Yes|Yes|
| SqueezeNet1_1 | Yes |Yes|Yes|
| VGG11_bn | Yes |Yes|Yes|
| WideResNet50 | Yes |Yes|Yes|
| Xception | Yes |Yes|Yes|

### NLP

| Model Name| PyTorch | TensorFlow | Keras |
| -- | -- | -- | -- |
| Bert | Yes |Yes|NO|
| Embedding | Yes |Yes|Yes|
| BiDirectionRNN | Yes |NO|Yes|
| RNN | Yes |NO|Yes|
| GRU | Yes |NO|Yes|
| LSTM | Yes |NO|Yes|

### Recommender

| Model Name| PyTorch | TensorFlow | Keras |
| -- | -- | -- | -- |
| DeepFM | Yes | Yes |Yes |
| DLRM| Yes |Yes |Yes |

## Operators

### PyTorch

| Op Name | Support |
| -- | -- |
| aten::abs | Yes |
| aten::acos | Yes |
| aten::acosh | Yes |
| aten::adaptive_avg_pool2d | Yes |
| aten::adaptive_avg_pool3d | Yes |
| aten::adaptive_max_pool2d | Yes |
| aten::adaptive_max_pool3d | Yes |
| aten::add | Yes |
| aten::addmm | Yes |
| aten::asin | Yes |
| aten::asinh | Yes |
| aten::atan | Yes |
| aten::atanh | Yes |
| aten::avg_pool2d | Yes |
| aten::avg_pool3d | Yes |
| aten::batch_norm | Yes |
| aten::bmm | Yes |
| aten::Bool | Yes |
| aten::cat | Yes |
| aten::ceil | Yes |
| aten::chunk | Yes |
| aten::clamp | Yes |
| aten::constant_pad_nd | Yes |
| aten::contiguous | Yes |
| aten::convolution | Yes |
| aten::cos | Yes |
| aten::cosh | Yes |
| aten::detach | Yes |
| aten::device | Yes |
| aten::div | Yes |
| aten::dropout | Yes |
| aten::elu | Yes |
| aten::embedding_bag | Yes |
| aten::exp | Yes |
| aten::expand | Yes |
| aten::expand_as | Yes |
| aten::feature_dropout | Yes |
| aten::flatten | Yes |
| aten::Float | Yes |
| aten::floor | Yes |
| aten::floordiv | Yes |
| aten::grid_sampler | Yes |
| aten::gru | Yes |
| aten::hardtanh | Yes |
| aten::index | Yes |
| aten::instance_norm | Yes |
| aten::Int | Yes |
| aten::layer_norm | Yes |
| aten::leaky_relu | Yes |
| aten::linear | Yes |
| aten::log | Yes |
| aten::lstm | Yes |
| aten::max | Yes |
| aten::max_pool2d | Yes |
| aten::max_pool3d | Yes |
| aten::mean | Yes |
| aten::min | Yes |
| aten::mul | Yes |
| aten::neg | Yes |
| aten::norm | Yes |
| aten::permute | Yes |
| aten::pow | Yes |
| aten::prelu | Yes |
| aten::reciprocal | Yes |
| aten::reflection_pad2d | Yes |
| aten::relu | Yes |
| aten::reshape | Yes |
| aten::rnn_relu | Yes |
| aten::rnn_tanh | Yes |
| aten::rsub | Yes |
| aten::select | Yes |
| aten::selu | Yes |
| aten::sigmoid | Yes |
| aten::sin | Yes |
| aten::sinh | Yes |
| aten::size | Yes |
| aten::slice | Yes |
| aten::softmax | Yes |
| aten::softplus | Yes |
| aten::split | Yes |
| aten::split_with_sizes | Yes |
| aten::sqrt | Yes |
| aten::squeeze | Yes |
| aten::stack | Yes |
| aten::std | Yes |
| aten::sub | Yes |
| aten::sum | Yes |
| aten::t | Yes |
| aten::tan | Yes |
| aten::tanh | Yes |
| aten::to | Yes |
| aten::transpose | Yes |
| aten::unsqueeze | Yes |
| aten::upsample_bilinear2d | Yes |
| aten::upsample_nearest2d | Yes |
| aten::var | Yes |
| aten::view | Yes |

### TensorFlow

| Op Name | Support |
| -- | -- |
| Add | Yes |
| AddV2 | Yes |
| AvgPool | Yes |
| Bert | Yes |
| BiasAdd | Yes |
| Cast | Yes |
| clamp | Yes |
| ConcatV2 | Yes |
| Conv2D | Yes |
| DepthwiseConv2dNative | Yes |
| Elu | Yes |
| ExpandDims | Yes |
| FusedBatchNormV3 | Yes |
| GatherV2 | Yes |
| Greater | Yes |
| Identity | Yes |
| IteratorGetNext | Yes |
| Less | Yes |
| MatMul | Yes |
| Max | Yes |
| MaxPool | Yes |
| Mean | Yes |
| Min | Yes |
| Mul | Yes |
| Pad | Yes |
| Pow | Yes |
| Range | Yes |
| RealDiv | Yes |
| Relu | Yes |
| Relu6 | Yes |
| Reshape | Yes |
| Sigmoid | Yes |
| Softmax | Yes |
| Square | Yes |
| Squeeze | Yes |
| StridedSlice | Yes |
| Sub | Yes |
| Sum | Yes |
| Tanh | Yes |
| Transpose | Yes |