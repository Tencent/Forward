import forward
import numpy as np


def TestForward(model_path, batch_size=1, seq_len=1, infer_mode='float32'):
    builder = forward.OnnxBuilder()

    input_ids = np.ones([batch_size, seq_len], dtype='int32')
    attention_mask = np.ones([batch_size, seq_len], dtype='int32')
    segment_ids = np.ones([batch_size, seq_len], dtype='int32')

    builder.set_mode(infer_mode)
    onnx_engine = builder.build(model_path)

    need_save = True
    if need_save:
        engine_path = onnx_path + '.engine'

        # save engine
        onnx_engine.save(engine_path)

        # load saved engine
        onnx_engine = forward.OnnxEngine()
        onnx_engine.load(engine_path)

        inputs = [input_ids, attention_mask, segment_ids]
        outputs = onnx_engine.forward(inputs)
        print(outputs)


if __name__ == "__main__":
    onnx_path = 'bert.onnx'
    TestForward(onnx_path, batch_size=32, seq_len=128)
