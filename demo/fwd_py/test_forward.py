import torch
import forward

# 1. BUILD step: load TensorFlow-Bert model to build Forward engine
builder = forward.TorchBuilder()

infer_mode = 'float32'  #  float32 / float16 / int8_calib / int8
builder.set_mode(infer_mode)

dummy_inputs = torch.randn(1, 3, 224, 224) 
# build_with_name interface, input names can be viewed by model viewer like Netron
'''
dummy_inputs = {"input" : torch.randn(1, 3, 224, 224) } 
engine = builder.build_with_name('path/to/jit/module', dummy_inputs)
'''

engine = builder.build('path/to/jit/module', dummy_inputs)

need_save = True
if need_save:
    engine_path = 'path/to/engine'
    engine.save(engine_path)

    engine = forward.TorchEngine()
    engine.load(engine_path)

# 2. FORWARD step: do inference with Forward engine
inputs = torch.randn(1, 3, 224, 224).cuda() # network input tensors
# if inputs are CUDA tensors, then outputs are also CUDA tensors.
# if inputs are CPU tensors, then outputs are also CPU tensors.
outputs = engine.forward(inputs)

# forward_with_name interface
'''
outputs = engine.forward_with_name(dummy_inputs) # dict_type inputs
'''
