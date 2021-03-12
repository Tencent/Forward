# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import argparse
import textwrap
import os

'''
Description:
    convert .ckpt model to .pb model including weights and network information in one file.
功能说明：
    将 .ckpt 模型转化为单文件 .pb 模型文件，该文件同时包含权重和网络信息。
'''

def save_large_parameters(sess):
    blacklist = []

    for var in tf.global_variables():
        size = np.prod(var.shape, dtype=np.int64)
        if size > (1 << 25):
            print('The parameter for ', var.op.name,
                  ' is too large and will be stored in a seperated file')
            blacklist.append(var.op.name)
            # convert variables to numpy and save in bin files
            filename = var.op.name.replace('/', '-') + '.w'
            print('Saving large variable {} to: {}'.format(
                var.op.name, filename))
            if os.path.exists(filename):
                print('Data file already exists. Skip saving.')
                continue
            data = np.array(sess.run(var))
            data_shape = np.array(data.shape)
            data_shape[0] += 1  # add a row with all zero
            data.resize(data_shape)
            data.astype("float32").tofile(filename)
            print('Large variable {} saved to: {}'.format(
                var.op.name, filename))

    return blacklist


def save_model(sess, input_graph_def, output_node_names, blacklist):
    output_graph_def = convert_variables_to_constants(
        sess=sess,
        input_graph_def=input_graph_def,
        output_node_names=output_node_names,
        variable_names_blacklist=blacklist)
    print("%d ops in graph." % len(output_graph_def.node))

    if model_cnt:
        with tf.gfile.GFile(
                args.output_filename + '_' + str(model_cnt) + '.pb',
                "wb") as f:
            f.write(output_graph_def.SerializeToString())
    else:
        with tf.gfile.GFile(args.output_filename + '.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())


#input_checkpoint = 'models/model.ckpt-28045573'
#out_graph = 'news_fm.pb'

# freeze_graph(input_checkpoint,
#             out_graph,
#             ["Sigmoid_1", "Sigmoid_2", "Sigmoid_3", "Sigmoid_4", "Sigmoid_5"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tensorflow Checkpoint-PBModel Converter",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "checkpoint",
        type=str,
        help='Checkpoint file to convert. Example: model.ckpt-28045573')
    parser.add_argument(
        "output_filename",
        type=str,
        default="model",
        help='The name of output pb files. Suffix is not needed.')
    parser.add_argument(
        "output_opnames",
        type=str,
        help=textwrap.dedent(
            """The name of the tensorflows ops that produce the output you need in the graph.
Ops are seperated by ','. 
If you need to generate more than one PB Model with different outputs, use a single ':' to seperate them.
Example: 
Sigmoid_1,Sigmoid_2 # A model with output of Sigmoid_1 and Sigmoid_2 ops will be generated.
Sigmoid_1:Sigmoid_2,Sigmoid_3 # Two models will be generated. One with the output of Sigmoid_1, and the other with the output of the rest ops.
                        """))
    args = parser.parse_args()

    sess = tf.Session()
    saver = tf.train.import_meta_graph(args.checkpoint + '.meta',
                                       clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    saver.restore(sess, args.checkpoint)

    blacklist = save_large_parameters(sess)

    models = args.output_opnames.split(':')
    model_cnt = 0
    for model_nodes in models:
        output_node_names = model_nodes.split(',')
        if model_cnt:
            output_graph_name = args.output_filename + '_' + str(
                model_cnt) + '.pb'
        else:
            output_graph_name = args.output_filename + '.pb'
        save_model(sess, input_graph_def, output_node_names, blacklist)
        model_cnt += 1

    print("%d models created in total." % model_cnt)
