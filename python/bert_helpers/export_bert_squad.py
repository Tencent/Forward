# -*- coding: utf-8 -*-
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
#
# ╔════════════════════════════════════════════════════════════════════════════════════════╗
# ║──█████████╗───███████╗───████████╗───██╗──────██╗───███████╗───████████╗───████████╗───║
# ║──██╔══════╝──██╔════██╗──██╔════██╗──██║──────██║──██╔════██╗──██╔════██╗──██╔════██╗──║
# ║──████████╗───██║────██║──████████╔╝──██║──█╗──██║──█████████║──████████╔╝──██║────██║──║
# ║──██╔═════╝───██║────██║──██╔════██╗──██║█████╗██║──██╔════██║──██╔════██╗──██║────██║──║
# ║──██║─────────╚███████╔╝──██║────██║──╚████╔████╔╝──██║────██║──██║────██║──████████╔╝──║
# ║──╚═╝──────────╚══════╝───╚═╝────╚═╝───╚═══╝╚═══╝───╚═╝────╚═╝──╚═╝────╚═╝──╚═══════╝───║
# ╚════════════════════════════════════════════════════════════════════════════════════════╝
#
# Authors: Aster JIAN (asterjian@qq.com)
#          Yzx (yzxyzxyzx777@outlook.com)
#          Ao LI (346950981@qq.com)
#          Paul LU (lujq96@gmail.com)
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.keras import backend as K
import modeling

K.set_learning_phase(0)


def build_and_restore_model(init_checkpoint, bert_config_file):

    input_ids = tf.placeholder(tf.int32, shape=(None, None), name='input_ids')
    input_mask = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='input_mask')
    segment_ids = tf.placeholder(tf.int32,
                                 shape=(None, None),
                                 name='segment_ids')

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        # embedding=None,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        # task_ids=None,
        use_one_hot_embeddings=True,
        scope="bert")

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    #batch_size = final_hidden_shape[0]
    #seq_length = final_hidden_shape[1]
    #hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, bert_config.hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("cls/squad/output_bias", [2],
                                  initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [-1, bert_config.hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    #logits = tf.reshape(logits, [batch_size, seq_length, 2])
    #logits = tf.transpose(logits, [2, 0, 1])

    # unstacked_logits = tf.unstack(logits, axis=0)

    # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # return (start_logits, end_logits)
    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    return tvars


def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    print(output_names)
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name
                for v in tf.global_variables()).difference(keep_var_names
                                                           or []))
        output_names = output_names or []
        #         output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                print(node.name)
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names,
                                                      freeze_var_names)
        return frozen_graph
    # export bert model


def main():
    bert_dir = "bert_base_squad2"

    init_checkpoint = os.path.join(bert_dir, "model.ckpt")  # 模型地址
    tiny_bert_config_file = os.path.join(bert_dir,
                                         "bert_config.json")  # 配置文件地址

    config = tf.ConfigProto(intra_op_parallelism_threads=8,
                            inter_op_parallelism_threads=8)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        model_params = build_and_restore_model(init_checkpoint,
                                               tiny_bert_config_file)
        saver = tf.train.Saver()
        # 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, init_checkpoint)

        frozen_graph = freeze_session(sess, output_names=["BiasAdd"])
        # Save
        tf.train.write_graph(frozen_graph,
                             ".",
                             "bert_base_squad2.pb",
                             as_text=False)


if __name__ == "__main__":
    main()
