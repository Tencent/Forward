# Models & Operators

----

- [Models & Operators](#models--operators)
  - [Models](#models)
    - [CV](#cv)
    - [NLP](#nlp)
    - [Recommender](#recommender)
  - [Operators](#operators)
    - [PyTorch](#pytorch)
    - [TensorFlow](#tensorflow)

----

> Update on Release v1.2.4

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
| DeepFM | NO | Yes |Yes |
| DLRM| NO |Yes |Yes |

## Operators

### PyTorch

> Support 138 torch Operators

| Op Name |
|--|
|aten::_convolution|
|aten::abs|
|aten::abs_|
|aten::acos|
|aten::acos_|
|aten::acosh|
|aten::acosh_|
|aten::adaptive_avg_pool2d|
|aten::adaptive_avg_pool3d|
|aten::adaptive_max_pool2d|
|aten::adaptive_max_pool3d|
|aten::add|
|aten::add_|
|aten::addmm|
|aten::arange|
|aten::asin|
|aten::asin_|
|aten::asinh|
|aten::asinh_|
|aten::atan|
|aten::atan_|
|aten::atanh|
|aten::atanh_|
|aten::avg_pool2d|
|aten::avg_pool3d|
|aten::batch_norm|
|aten::bmm|
|aten::cat|
|aten::ceil|
|aten::ceil_|
|aten::chunk|
|aten::clamp|
|aten::clamp_|
|aten::constant_pad_nd|
|aten::contiguous|
|aten::cos|
|aten::cos_|
|aten::cosh|
|aten::cosh_|
|aten::detach|
|aten::device|
|aten::div|
|aten::div_|
|aten::dropout|
|aten::dropout_|
|aten::elu|
|aten::embedding|
|aten::embedding_bag|
|aten::eq|
|aten::erf|
|aten::erf_|
|aten::exp|
|aten::exp_|
|aten::expand|
|aten::expand_as|
|aten::feature_dropout|
|aten::flatten|
|aten::floor|
|aten::floor_|
|aten::floordiv|
|aten::gelu|
|aten::grid_sampler|
|aten::gru|
|aten::gt|
|aten::hardtanh|
|aten::hardtanh_|
|aten::index|
|aten::instance_norm|
|aten::layer_norm|
|aten::leaky_relu|
|aten::leaky_relu_|
|aten::linear|
|aten::log|
|aten::log_|
|aten::lrn|
|aten::lstm|
|aten::lt|
|aten::matmul|
|aten::max|
|aten::max_pool2d|
|aten::max_pool3d|
|aten::mean|
|aten::min|
|aten::mul|
|aten::mul_|
|aten::neg|
|aten::neg_|
|aten::norm|
|aten::ones|
|aten::permute|
|aten::pixel_shuffle|
|aten::pow|
|aten::prelu|
|aten::reciprocal|
|aten::reciprocal_|
|aten::reflection_pad2d|
|aten::relu|
|aten::relu_|
|aten::repeat|
|aten::reshape|
|aten::rnn_relu|
|aten::rnn_tanh|
|aten::rsub|
|aten::select|
|aten::selu|
|aten::sigmoid|
|aten::sin|
|aten::sin_|
|aten::sinh|
|aten::sinh_|
|aten::slice|
|aten::softmax|
|aten::softplus|
|aten::split|
|aten::split_with_sizes|
|aten::sqrt|
|aten::sqrt_|
|aten::squeeze|
|aten::stack|
|aten::std|
|aten::sub|
|aten::sub_|
|aten::sum|
|aten::tan|
|aten::tan_|
|aten::tanh|
|aten::to|
|aten::transpose|
|aten::unsqueeze|
|aten::unsqueeze_|
|aten::upsample_bilinear2d|
|aten::upsample_bilinear2d|
|aten::upsample_nearest2d|
|aten::var|
|aten::view|
|aten::zeros|
|Adaptive Layer-Instance Normalization|
|BERT|

### TensorFlow

> Support 44 TF Operators

|Op Name|
|--|
|Add|
|AddV2|
|AvgPool|
|BERT|
|Cast|
|Clamp|
|ConcatV2|
|Conv1D|
|Conv2D|
|Dense|
|DepthToSpace|
|DepthwiseConv2dNative|
|Elu|
|EmbeddingBag|
|ExpandDims|
|FusedBatchNorm|
|FusedBatchNormV3|
|GatherV2|
|Greater|
|Identity|
|Less|
|MatMul|
|Max|
|MaxPool|
|Mean|
|Min|
|Mul|
|Pad|
|Pow|
|Range|
|RealDiv|
|Relu|
|Reshape|
|Select|
|Sigmoid|
|Softmax|
|Split|
|Square|
|Squeeze|
|StridedSlice|
|Sub|
|Sum|
|Tanh|
|Transpose|