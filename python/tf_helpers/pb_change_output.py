# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import argparse
import textwrap
from google.protobuf import text_format
import os

'''
Description:
    modify .pb file: change the OUTPUT nodes according to given output names.
功能说明：
    修改 pb 模型，输出节点改为指定输出节点。
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
        with tf.gfile.GFile(args.output_pb + '_' + str(model_cnt) + '.pb',
                            "wb") as f:
            f.write(output_graph_def.SerializeToString())
    else:
        with tf.gfile.GFile(args.output_pb + '.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tensorflow Checkpoint-PBModel Converter",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "input_pb",
        type=str,
        default="model_in",
        help='The name of input pb files. Suffix is not needed.')
    parser.add_argument(
        "output_pb",
        type=str,
        default="model_out",
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
    if os.path.exists(args.input_pb + ".pb"):
        print("Loading " + args.input_pb + ".pb")
        with gfile.FastGFile(args.input_pb + ".pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
    elif os.path.exists(args.input_pb + ".pbtxt"):
        print("Loading " + args.input_pb + ".pbtxt")
        # Let's read our pbtxt file into a Graph protobuf
        with open(args.input_pb + ".pbtxt", "r") as f:
            graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())
            sess.graph.as_default()
            tf.import_graph_def(graph_protobuf, name='')  # 导入计算图
    else:
        print("ProtoBuf not found.")
        exit(1)

    blacklist = save_large_parameters(sess)

    sess.run(tf.global_variables_initializer())
    input_graph_def = tf.get_default_graph().as_graph_def()

    models = args.output_opnames.split(':')
    model_cnt = 0
    for model_nodes in models:
        output_node_names = model_nodes.split(',')
        if model_cnt:
            output_graph_name = args.output_pb + '_' + str(model_cnt) + '.pb'
        else:
            output_graph_name = args.output_pb + '.pb'
        save_model(sess, input_graph_def, output_node_names, blacklist)
        model_cnt += 1

    print("%d models created in total." % model_cnt)
