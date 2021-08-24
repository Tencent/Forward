import forward
import numpy as np

# 1. Build Engine
model_path='frozen_bert.pb'
engine_path='bert.engine'
batch_size = 16
seq_len = 128
infer_mode = 'float32'

builder = forward.TfBuilder()
dummy_input = {"input_ids" : np.ones([batch_size , seq_len], dtype='int32'),
               "input_mask" : np.ones([batch_size , seq_len], dtype='int32'),
               "segment_ids" : np.ones([batch_size , seq_len], dtype='int32')}

builder.set_mode(infer_mode)
tf_engine = builder.build(model_path, dummy_input)

need_save = True
if need_save:
    # save engine
    tf_engine.save(engine_path)

    # load saved engine
    tf_engine = forward.TfEngine()
    tf_engine.load(engine_path)

inputs = dummy_input
outputs = tf_engine.forward(inputs)
print(outputs)
