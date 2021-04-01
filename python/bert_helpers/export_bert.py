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
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names,
                                                      freeze_var_names)
        return frozen_graph
    # export bert model


def main():
    bert_dir = "tiny_bert"
    pathname = os.path.join(bert_dir, "bert_model.ckpt")  # 模型地址
    bert_config = modeling.BertConfig.from_json_file(
        os.path.join(bert_dir, "bert_config.json"))  # 配置文件地址

    configsession = tf.ConfigProto()
    configsession.gpu_options.allow_growth = True
    sess = tf.Session(config=configsession)
    input_ids = tf.placeholder(shape=[None, 64],
                               dtype=tf.int32,
                               name="input_ids")
    input_mask = tf.placeholder(shape=[None, 64],
                                dtype=tf.int32,
                                name="input_mask")
    segment_ids = tf.placeholder(shape=[None, 64],
                                 dtype=tf.int32,
                                 name="segment_ids")

    with sess.as_default():
        model = modeling.BertModel(config=bert_config,
                                   is_training=False,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=True)
        saver = tf.train.Saver()
        # 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, pathname)

        # frozen_graph = freeze_session(sess, output_names=['bert/encoder/Reshape_3'])
        frozen_graph = freeze_session(sess,
                                      output_names=['bert/pooler/dense/Tanh'])
        # Save
        tf.train.write_graph(frozen_graph, ".", "tiny_bert.pb", as_text=False)


if __name__ == "__main__":
    main()
