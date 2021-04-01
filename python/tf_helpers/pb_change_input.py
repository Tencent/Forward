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
import tensorflow as tf
'''
Description:
    modify .pb file: change the INPUT nodes of IteratorGetNext type into INPUT nodes of PlaceHolder type.
功能说明：
    修改 pb 模型，将一些特殊输入节点（如 IteratorGetNext）替换为 placeholder 输入。
'''

input_model_path = 'new_bert.pb'
out_model_path = "changed_new_bert.pb"

# 加载 pb 模型
with tf.gfile.GFile(input_model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
graph = tf.get_default_graph()

# 设置想要使用的 Placeholder 输入以及合适的名称，注意：数据类型要匹配。
iunput_ids = tf.placeholder(shape=(None, None),
                            dtype='int64',
                            name='input_ids')
input_mask = tf.placeholder(shape=(None, None),
                            dtype='int64',
                            name='input_mask')
# input_mask_float = tf.placeholder(shape=(None, None), dtype='float32', name='input_mask_float')
segment_ids = tf.placeholder(shape=(None, None),
                             dtype='int64',
                             name='segment_ids')

# 将 placeholder 映射到要替换的输入 Tensor，名称形式一般为  'OperationName:Index'
input_map = {
    "split:0": iunput_ids,
    'split_1:0': input_mask,
    'split_2:0': segment_ids
}
# input_map = {"split:0": iunput_ids, 'split_1:0': input_mask, 'split_2:0': input_mask_float, 'split_3:0': segment_ids}

tf.import_graph_def(graph_def, name='', input_map=input_map)

# 保存为新的 pb 模型
tf.train.write_graph(graph, ".", out_model_path, as_text=False)
