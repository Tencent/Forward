# Libraries
> Dynamic libraries listed below should be copied here from cmake-build directories.
- libfwd_torch.so [optional]: Used for build engine from TorchScript Module. Also used for doing inference with Torch Tensors.
- libfwd_tf.so [optional]: Used for build engine from Tensorflow pb model. Also used for doing inference with TF_Tensors.
- libfwd_onnx.so [optional]: Used for build engine from ONNX model. Also used for doing inference with Forward Tensors.
- libtrt_engine.so [required] : Used alone for doing inference with Fwd_Tensors.
