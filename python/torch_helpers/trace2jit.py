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
