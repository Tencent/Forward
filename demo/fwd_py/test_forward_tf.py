import tensorflow
import forward
import numpy as np

# 1. BUILD step: load TensorFlow-Bert model to build Forward engine
builder = forward.TfBuilder()

infer_mode = 'float32'  #  float32 / float16 / int8_calib / int8
builder.set_mode(infer_mode)

dummy_inputs = np.random.random((1, 3, 224, 224)).astype('float32')
feed_dict = {'input_11':dummy_inputs}
# build_with_name interface, input names can be viewed by model viewer like Netron

model_path = 'softmax.pb'
engine = builder.build(model_path, feed_dict)

need_save = True
if need_save:
    engine_path = model_path + '.engine'
    engine.save(engine_path)

    engine = forward.TfEngine()
    engine.load(engine_path)

# 2. FORWARD step: do inference with Forward engine
# if inputs are CUDA tensors, then outputs are also CUDA tensors.
# if inputs are CPU tensors, then outputs are also CPU tensors.
outputs = engine.forward(feed_dict)
print(outputs)
