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
#          Paul LU (lujq96@gmail.com)import torch
import torch
import torchvision.models as models

'''
Description:
    convert torch module to JIT TracedModule.
功能说明：
    将torch 模型转化为 JIT TracedModule。
'''

def TracedModelFactory(file_name, traced_model):
    traced_model.save(file_name)
    traced_model = torch.jit.load(file_name)
    print("filename : ", file_name)
    print(traced_model.graph)

    
if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224) # dummy_input is customized by user
    model = models.resnet18(pretrained=True)  # model is customized by user

    model = model.cpu().eval()
    traced_model = torch.jit.trace(model, dummy_input)

    model_name = 'model_name'   # model_name is customized by user
    TracedModelFactory(model_name + '.pth', traced_model)
