import forward
import numpy as np

# 1. BUILD step: load ResNet50 model to build Forward engine
builder = forward.OnnxBuilder()

infer_mode = 'float32'  #  float32 / float16 / int8_calib / int8
builder.set_mode(infer_mode)

model_path = 'softmax.onnx'
engine = builder.build(model_path)

need_save = True
if need_save:
    engine_path = model_path + ".engine"
    engine.save(engine_path)

    engine = forward.OnnxEngine()
    engine.load(engine_path)

# 2. FORWARD step: do inference with Forward engine
# Even if the model has a single input, convert the NumPy array in the form of a list as so 
# to avoid Pybind11 mistakenly parsing the single NumPy array into multiple lower-dimension
# NumPy arrays.
inputs = np.random.rand(1, 3, 224, 224).astype('float32')
outputs = engine.forward([inputs])
print(outputs)
