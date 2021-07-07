# Tool and Test

## Auxiliary Tools

In order to analyze the network, we provide some auxiliary tools which will output useful information during the build and inference steps. The relevant code includes:

- `TrtForward::DumpNetwork`: print the `TensorRT` network, including the layer name, the dimensions and types of the inputs and outputs, etc. To be noticed, the type information might not be precise and this is a known bug related to `TensorRT`. This function can be called inside the `TrtForward::Build` function and the user has the choice to enable or disable it.
- `TrtCommon::SimpleProfiler`: print the inference time in each layer. To use this function, the user needs to include the macro `TRT_INFER_ENABLE_PROFILING`, which means the user needs to include the `ENABLE_PROFILING` option in the build step. Forward will print the inference time once `TrtForward` finishes inference.

## Unit Tests

For most of the conversion of individual nodes, we provide unit tests to verify the correctness of its conversion. These tests are under the [unit_test](../../../source/unit_test) folder.

| File Name | Content |
| :-----| :---- |
| `test_<platform>_nodes.h` | Verify the correctness of the conversion of individual nodes under the corresponding platform |

In `unit_test_<platform>_helper.h`, we provide the method `Test<Platform>Inference` and users can continue to add unit tests according to this method.

## Integration Tests

For those common models in CV, Bert, and Recommender fields, we also provide corresponding tests under the [unit_test](../../../source/unit_test) folder to verify the correctness of the model conversion.

| File Name | Content |
| :-----| :---- |
| `test_<platform>_vision.h` | Verify the correctness of CV-related model conversion under the corresponding platform |
| `test_<platform>_bert.h` | Verify the correctness of Bert-related model conversion under the corresponding platform |
| `test_torch_dlrm.h.h` | Verify the correctness of DLRM model conversion under PyTorch platform |
| `test_tf_recommender.h` | Verify the correctness of recommender model conversion under TensorFlow platform |

## Performance Tests

For the performance metrics of the CV-related models, we have verified their performance through performance tests. In `unit_test_<platform>_helper.h`, we provide the method `Test<Platform>Time` and users can continue to add unit tests according to this method.