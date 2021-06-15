#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn.functional as F
from collections import OrderedDict

def TracedModelFactory(file_name, traced_model):
    traced_model.save(file_name)
    traced_model = torch.jit.load(file_name)
    print("filename : ", file_name)
    print(traced_model.graph)

class ArithmeticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b):
        a1 = torch.add(a, b)
        b1 = torch.rsub(a, b)
        c1 = torch.sub(a, b)
        d1 = torch.mul(a, b)
        return a1, b1, c1, d1
    
a = torch.randn(4)
b = torch.randn(4)
model = ArithmeticModule()
model.eval()
traced_model = torch.jit.trace(model, (a, b))

TracedModelFactory("arithmetic.pth", traced_model)

class ActivationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
    
    def forward(self, x):
        t = self.tanh(x)
        s = self.sigmoid(x)
        r = self.relu(x)
        lr = self.leaky_relu(x)
        return t,s,r,lr
        
    
a = torch.randn(4)
model = ActivationModule()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("activation.pth", traced_model)

class AdaILN(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out

num_features = 7
model = AdaILN(num_features)
model.eval()

input = torch.randn(1, num_features, 23, 34)
gamma = torch.randn(1, num_features)
beta = torch.randn(1, num_features)
traced_model = torch.jit.trace(model, (input, gamma, beta))
TracedModelFactory("AdaILN.pth", traced_model)

class AddmmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, M, mat1, mat2):
        return torch.addmm(M, mat1, mat2)
    
M = torch.randn(2, 3)
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
model = AddmmModule()
model.eval()
traced_model = torch.jit.trace(model, (M, mat1, mat2))

TracedModelFactory("addmm.pth", traced_model)

class AdaptivePooling2dModule(torch.nn.Module):
    def __init__(self):
        super(AdaptivePooling2dModule, self).__init__()
        self.adaptive_max_pool2d_1 = torch.nn.AdaptiveMaxPool2d((5,7))
        self.adaptive_max_pool2d_2 = torch.nn.AdaptiveMaxPool2d((None, 7))
        self.adaptive_avg_pool2d_1 = torch.nn.AdaptiveAvgPool2d((5,7))
        self.adaptive_avg_pool2d_2 = torch.nn.AdaptiveAvgPool2d((None, 7))
    
    def forward(self, x):
        y1 = self.adaptive_max_pool2d_1(x)
        y2 = self.adaptive_max_pool2d_2(x)
        y3 = self.adaptive_avg_pool2d_1(x)
        y4 = self.adaptive_avg_pool2d_2(x)
        return y1, y2, y3, y4

dummy_input = torch.randn(1, 64, 10, 9)
model = AdaptivePooling2dModule()
model.eval()
y1, y2, y3, y4 = model(dummy_input)

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("adaptive_pooling_2d.pth", traced_model)

class AdaptivePooling3dModule(torch.nn.Module):
    def __init__(self):
        super(AdaptivePooling3dModule, self).__init__()
        self.adaptive_max_pool3d_1 = torch.nn.AdaptiveMaxPool3d((7, 8, 9))
        self.adaptive_max_pool3d_2 = torch.nn.AdaptiveMaxPool3d((7, None, 8))
        self.adaptive_avg_pool3d_1 = torch.nn.AdaptiveAvgPool3d((7, 8, 9))
        self.adaptive_avg_pool3d_2 = torch.nn.AdaptiveAvgPool3d((7, None, 8))
    
    def forward(self, x):
        y1 = self.adaptive_max_pool3d_1(x)
        y2 = self.adaptive_max_pool3d_2(x)
        y3 = self.adaptive_avg_pool3d_1(x)
        y4 = self.adaptive_avg_pool3d_2(x)
        return y1, y2, y3, y4

dummy_input = torch.randn(1, 64, 8, 9, 10)
model = AdaptivePooling3dModule()
model.eval()
y1, y2, y3, y4 = model(dummy_input)

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("adaptive_pooling_3d.pth", traced_model)

# torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
class BatchNormModule(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels, affine=False)
        self.batch_norm3 = torch.nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
    
    def forward(self, x):
        return self.batch_norm1(x), self.batch_norm2(x), self.batch_norm3(x)
    
state_dict = OrderedDict([('batch_norm1.weight', torch.randn(3)), ('batch_norm1.bias', torch.randn(3)),
                         ('batch_norm2.weight', torch.randn(3)), ('batch_norm2.bias', torch.randn(3)),
                         ('batch_norm3.weight', torch.randn(3)), ('batch_norm3.bias', torch.randn(3))])

dummy_input = torch.randn((2, 3, 32, 32))
model = BatchNormModule(3)
model.load_state_dict(state_dict, strict=False)
model(dummy_input)
model.eval()

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("batch_norm_2d.pth", traced_model)

class BmmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        y1 = torch.bmm(x, y)
        y2 = torch.bmm(y.permute(0,2,1), x.transpose(1,2))
        return y1, y2
    
dummy_input1 = torch.randn((3, 4, 5))
dummy_input2 = torch.randn((3, 5, 6))
model = BmmModule()
model.eval()
traced_model = torch.jit.trace(model, (dummy_input1, dummy_input2))

TracedModelFactory("bmm.pth", traced_model)

class ConstantPad2dModule(torch.nn.Module):
    def __init__(self, dims, val):
        super().__init__()
        self.constant_pad = torch.nn.ConstantPad2d(dims, val)
    
    def forward(self, x):
        return self.constant_pad(x), torch.nn.ConstantPad2d((1, 1, 1, 1), 0)(x.permute(2, 0, 1, 3))
    
dummy_input1 = torch.randn(1, 2, 2, 4)
model = ConstantPad2dModule((3, 0, 2, 1), 3.5)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("constant_pad_2d.pth", traced_model)

class ConstantPad3dModule(torch.nn.Module):
    def __init__(self, dims, val):
        super().__init__()
        self.constant_pad = torch.nn.ConstantPad3d(dims, val)
    
    def forward(self, x):
        return self.constant_pad(x), torch.nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 0)(x.permute(2, 0, 1, 3, 4))
    
dummy_input1 = torch.randn(1, 14, 34, 12, 45)
model = ConstantPad3dModule((5, 7, 3, 0, 2, 1), 1.7)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("constant_pad_3d.pth", traced_model)

class ClampModule(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, x):
        return torch.clamp(x, self.min, self.max)
    
dummy_input1 = torch.randn((3, 4, 5))
dummy_min = -1.
dummy_max = 1.
model = ClampModule(dummy_min, dummy_max)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("clamp.pth", traced_model)

class CatModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.cat([x, x], dim = self.dim)
    
dummy_input = torch.randn((2, 3, 32, 32))
model = CatModule(1)
model.eval()
traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("cat.pth", traced_model)

class Conv2dModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    def forward(self, x):
        return self.conv1(x), self.conv2(x)
    
dummy_input1 = torch.randn((2, 3, 32, 64))
model = Conv2dModule(3, 6, 3, 2, 1)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("conv2d.pth", traced_model)

#torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
class ConvTranspose2dModule(torch.nn.Module):
    def __init__(self):
        super(ConvTranspose2dModule, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 3, (3, 5), 2, 1, 1, bias=True),
            torch.nn.ConvTranspose2d(3, 2, (2, 1), 1, 0, bias=False))
    
    def forward(self, x):
        return self.seq(x)
    
dummy_input1 = torch.randn((1, 8, 32, 64))
model = ConvTranspose2dModule()
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("deconv2d.pth", traced_model)

import numpy as np
class EmbeddingBagModule(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.EE = torch.nn.EmbeddingBag(n, m, mode="sum", sparse = True)
        W = np.random.uniform(
            low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
        ).astype(np.float32)
        # approach 1
        self.EE.weight.data = torch.tensor(W, requires_grad=True)
        
    def forward(self, index, offset):
        V = self.EE(index, offset)
        return V
    
model = EmbeddingBagModule(5,4)
model.eval()
traced_model = torch.jit.trace(model, (torch.tensor([1,0,3,1,4]), torch.tensor([0,2])))
TracedModelFactory("embedding_bag.pth", traced_model)

class ExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # y1 = x.expand(1,3,2)
        w = torch.Tensor([[[1.2],[3.4],[4.1]]])
        w = w.expand(1,3,2)
        return x + w
    
x = torch.randn(1,3,2)
model = ExpandModule()
model.eval()
traced_model = torch.jit.trace(model, (x))

TracedModelFactory("expand.pth", traced_model)

class FloorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.floor(x)
        
    
a = torch.randn(4)
model = FloorModule()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("floor.pth", traced_model)

class FullyConnectedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 100)
        self.fc2 = torch.nn.Linear(100, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    
x = torch.randn((1,10))
model = FullyConnectedModule()
model.eval()
traced_model = torch.jit.trace(model, x)

TracedModelFactory("fully_connected.pth", traced_model)

class GeluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.functional.gelu
    
    def forward(self, x):
        return self.gelu(x)
    
x = torch.randn(4)
model = GeluModule()
model.eval()
traced_model = torch.jit.trace(model, (x))

TracedModelFactory("gelu.pth", traced_model)

class GRUModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(GRUModule, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.gru = torch.nn.GRU(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x, h_0):
        h_t = h_0.permute(1,0,2)
        out3, _ = self.gru(x, h_t)
        #out3 = out3[:, -1, :]
        #out3 = self.classifier(out3)
        return out3
x =  torch.randn(1, 28, 28)
model = GRUModule(28, 128, 2, 10)
model.eval()
traced_model = torch.jit.trace(model, (x,torch.randn(1, 2, 128)))

TracedModelFactory("gru.pth", traced_model)

import torch.nn.functional as F
class GridSamplerBilinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, T):
        y_zeros = F.grid_sample(x, T, mode='bilinear', padding_mode='zeros', align_corners=True)
        y_border = F.grid_sample(x, T, mode='bilinear', padding_mode='border', align_corners=False)
        y_reflection = F.grid_sample(x, T, mode='bilinear', padding_mode='reflection', align_corners=True)
        return (y_zeros, y_border, y_reflection)
    
x = torch.randn(1, 3, 5, 7)
T = torch.randn(1, 5, 10, 2) # interpolation in width
model = GridSamplerBilinearModule()
model.eval()
traced_model = torch.jit.trace(model, (x, T))

TracedModelFactory("grid_sampler_bilinear.pth", traced_model)

class GridSamplerNearestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, T):
        y_zeros = F.grid_sample(x, T, mode='nearest', padding_mode='zeros', align_corners=True)
        y_border = F.grid_sample(x, T, mode='nearest', padding_mode='border', align_corners=False)
        y_reflection = F.grid_sample(x, T, mode='nearest', padding_mode='reflection', align_corners=False)
        return (y_zeros, y_border, y_reflection)
x = torch.randn(1, 3, 5, 7)
T = torch.randn(1, 6, 7, 2) # interpolation in height
model = GridSamplerNearestModule()
model.eval()
traced_model = torch.jit.trace(model, (x, T))

TracedModelFactory("grid_sampler_nearest.pth", traced_model)

class IndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.index_1 = [1,5,3,2]
        self.index_2 = [2,3,1,1]
        
    def forward(self, X):
        V = X[:, self.index_1, self.index_2]
        return V
    
model = IndexModule()
model.eval()
traced_model = torch.jit.trace(model, (torch.randn(1,10,10)))
TracedModelFactory("index.pth", traced_model)

class ILN(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = torch.nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.2)
        self.gamma.data.fill_(1.3)
        self.beta.data.fill_(-1.1)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out
    
num_features = 7
model = ILN(num_features)
model.eval()

input = torch.randn(1, num_features, 23, 34)
traced_model = torch.jit.trace(model, (input))
TracedModelFactory("ILN.pth", traced_model)

class InplaceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.leaky_relu = torch.nn.LeakyReLU(inplace=True)
        self.dropout = torch.nn.Dropout(inplace=True)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.leaky_relu(x)
        x = torch.abs_(x)
        # x = self.dropout(x)
        return x
    
x = torch.randn(4)
model = InplaceModule()
model.eval()
traced_model = torch.jit.trace(model, x)

TracedModelFactory("inplace.pth", traced_model)

from collections import OrderedDict

class InstanceNorm2dModule(torch.nn.Module):
    def __init__(self, out_channels, affine, track):
        super().__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(out_channels, affine=affine, track_running_stats=track)
    
    def forward(self, x):
        return self.instance_norm(x)

state_dict = OrderedDict([('instance_norm.weight', torch.randn(3)), ('instance_norm.bias', torch.randn(3))])

dummy_input1 = torch.randn((2, 3, 32, 64))
model = InstanceNorm2dModule(3, True, True)
model.load_state_dict(state_dict, strict=False)
model(dummy_input1)
model(dummy_input1)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("instance_norm_aff_track.pth", traced_model)

model = InstanceNorm2dModule(3, True, False)
model.load_state_dict(state_dict, strict=False)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("instance_norm_aff.pth", traced_model)

model = InstanceNorm2dModule(3, False, True)
model.load_state_dict(state_dict, strict=False)
model(dummy_input1)
model(dummy_input1)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("instance_norm_track.pth", traced_model)

model = InstanceNorm2dModule(3, False, False)
model.load_state_dict(state_dict, strict=False)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("instance_norm.pth", traced_model)

class LrnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        result = F.local_response_norm(x, size=5)
        result *= 1.0
        return result
x = torch.randn(1, 3, 5, 7)
model = LrnModule()
model.eval()
traced_model = torch.jit.trace(model, x)

TracedModelFactory("lrn.pth", traced_model)

# torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y1 = torch.nn.LayerNorm(x.shape[1:], elementwise_affine=False)(x)
        #y2 = torch.nn.LayerNorm([14, 32], elementwise_affine=False)(x)
        #y3 = torch.nn.LayerNorm([14], elementwise_affine=False)(x)
        return y1, #y2, y3

dummy_input = torch.randn((2, 3, 14, 32))
model = LayerNormModule()
model.eval()

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("layer_norm.pth", traced_model)

class LayerNormModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(10)
    
    def forward(self, x):
        y1 = self.ln(x)
        return y1

dummy_input = torch.randn((20, 5, 10))
model = LayerNormModule2()
model.eval()

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("layer_norm_with_weights.pth", traced_model)

class LSTMModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMModule, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x, h_0, c_0):
        h_t = h_0.permute(1, 0, 2)
        c_t = c_0.permute(1, 0, 2)
        out1, _ = self.lstm(x, (h_t, c_t))
        #out1 = out1[:, -1, :]
        #out1 = self.classifier(out1)
        return out1
x =  torch.randn(1, 28, 28)
model = LSTMModule(28, 128, 2, 10)
model.eval()
traced_model = torch.jit.trace(model, (x, torch.randn(1, 2, 128), torch.randn(1, 2, 128)))

TracedModelFactory("lstm.pth", traced_model)

class LSTM2Module(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM2Module, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.lstm2 = torch.nn.LSTM(hidden_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x, h_0, c_0):
        h_t = h_0.permute(1, 0, 2)
        c_t = c_0.permute(1, 0, 2)
        
        out1, hc_1 = self.lstm(x, (h_t, c_t))
        out2, _ = self.lstm2(out1, hc_1)
        #out1 = out1[:, -1, :]
        #out1 = self.classifier(out1)
        return out2
x =  torch.randn(1, 28, 28)
model = LSTM2Module(28, 128, 2, 10)
model.eval()
traced_model = torch.jit.trace(model, (x, torch.randn(1, 2, 128), torch.randn(1, 2, 128)))

TracedModelFactory("lstm2.pth", traced_model)

class NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x/torch.norm(x, p=2, dim=1, keepdim=True)
        
    
a = torch.randn((1,32,1,1))
model = NormModule()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("norm.pth", traced_model)

class PermuteModule(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        y = x.permute(self.dims).contiguous()
        return y
        
    
a = torch.randn(2, 3, 5)
model = PermuteModule((2, 0, 1))
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("permute.pth", traced_model)

class PixelShuffleModule(torch.nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        y = self.pixel_shuffle(x)
        return y
        
    
a = torch.randn(1, 9, 24, 24)
model = PixelShuffleModule(3)
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("pixel_shuffle.pth", traced_model)

class PReluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu1 = torch.nn.PReLU(3, 0.11)
        self.prelu2 = torch.nn.PReLU()
    
    def forward(self, x):
        return self.prelu1(x), self.prelu2(x)
    
x = torch.randn(1, 3, 11, 13)
model = PReluModule()
model.eval()
traced_model = torch.jit.trace(model, x)

TracedModelFactory("prelu.pth", traced_model)

class Pooling2dModule(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Pooling2dModule, self).__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size, stride, padding)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size, stride, padding)
    
    def forward(self, x):
        y1 = self.max_pool2d(x)
        y2 = self.avg_pool2d(x)
        return y1, y2

dummy_input = torch.randn(1, 23, 54, 96)
model = Pooling2dModule(3, 2, 1)
model.eval()
y1, y2 = model(dummy_input)
print (y1.shape, y2.shape)

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("pooling_2d.pth", traced_model)

class Pooling3dModule(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Pooling3dModule, self).__init__()
        self.max_pool3d = torch.nn.MaxPool3d(kernel_size, stride, padding)
        self.avg_pool3d = torch.nn.AvgPool3d(kernel_size, stride, padding)

    def forward(self, x):
        y1 = self.max_pool3d(x)
        y2 = self.avg_pool3d(x)
        return y1, y2

dummy_input = torch.randn(1, 14, 23, 54, 96)
model = Pooling3dModule(3, 2, 1)
model.eval()
y1, y2 = model(dummy_input)
print (y1.shape, y2.shape)

traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("pooling_3d.pth", traced_model)

class ReflectionPad2dModule(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.reflection_pad = torch.nn.ReflectionPad2d(dims)
    
    def forward(self, x):
        return self.reflection_pad(x)
    
dummy_input1 = torch.randn((1, 1, 3, 3))
model = ReflectionPad2dModule((1, 1, 2, 0))
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("reflection_pad_2d.pth", traced_model)

class RNN0Module(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN0Module, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.rnn_tanh_bidirect = torch.nn.RNN(in_dim, hidden_dim, n_layer,
                            batch_first=True, bidirectional=True)
        self.classifier2 = torch.nn.Linear(hidden_dim*2, n_class)

    def forward(self, x):
        out0, _ = self.rnn_tanh_bidirect(x)
        #out0 = out0[:, -1, :]
        #out0 = self.classifier2(out0)
        return out0
    
x =  torch.randn(1, 28, 28)
model = RNN0Module(28, 128, 2, 10)
model.eval()
traced_model = torch.jit.trace(model, x)

TracedModelFactory("rnn_tanh_bidirectional.pth", traced_model)

class RNN1Module(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN1Module, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.rnn_relu = torch.nn.RNN(in_dim, hidden_dim, n_layer,
                            batch_first=True, nonlinearity='relu')
        self.classifier = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out2, _ = self.rnn_relu(x)
        #out2 = out2[:, -1, :]
        #out2 = self.classifier(out2)
        return out2
x =  torch.randn(1, 28, 28)
model = RNN1Module(28, 128, 2, 10)
model.eval()
traced_model = torch.jit.trace(model, x)

TracedModelFactory("rnn_relu.pth", traced_model)

class ReduceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = x.mean((2,3), False)
        x = x.min(1, False).values
        x = x.max(0, True).values
        x = x.sum()
        return x
    
a = torch.randn((32, 16, 45, 12))
model = ReduceModule()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("reduce.pth", traced_model)

class Reduce0Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = x.sum(0, False)
        return y
    
a = torch.randn((32, 16, 45, 12))
model = Reduce0Module()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("reduce_0.pth", traced_model)

class Reduce1Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = x.sum(1, False)
        return y
    
a = torch.randn((32, 16, 45, 12))
model = Reduce1Module()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("reduce_1.pth", traced_model)

class Reduce2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = x.sum(2, False)
        return y
    
a = torch.randn((32, 16, 45, 12))
model = Reduce2Module()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("reduce_2.pth", traced_model)

class Reduce3Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = x.sum(3, False)
        return y
    
a = torch.randn((32, 16, 45, 12))
model = Reduce3Module()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("reduce_3.pth", traced_model)

class RepeatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y1 = x.repeat(1,1)
        y2 = x.repeat(2,1)
        y3 = x.repeat(1,2)
        y4 = x.repeat(1,1,1)
        y5 = x.repeat(3,1,1)
        y6 = x.repeat(1,3,1)
        y7 = x.repeat(1,1,3)
        y8 = x.repeat(2,1,1,2)
        return (y1, y2, y3, y4, y5, y6, y7, y8)
    
dummy_input = torch.randn((2, 3))
model = RepeatModule()
model.eval()
traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("repeat.pth", traced_model)

class StackModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.stack([x, x], dim = self.dim)
    
dummy_input = torch.randn((2, 3, 32, 32))
model = StackModule(1)
model.eval()
traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("stack.pth", traced_model)

class SoftMaxModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=dim)
    
    def forward(self, x):
        return self.softmax(x)
    
dummy_input1 = torch.randn((3, 4, 5))
model = SoftMaxModule(-1)
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("softmax.pth", traced_model)

class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = x[:,:,2:20,3:-1:2]
        y1 = y[:, 2, :, 2: 10 : 2]
        z = x[:,:,:,0:10:3]
        k = x[:,2:3,3:5,4:20:2]
        k1 = k[:,:,1,:-1]
        
        return y1[:, 2:, :-1], z[:, 1:], k1[:, :, 0]
    
dummy_input1 = torch.randn((4, 64, 64, 64))
model = SliceModule()
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("slice.pth", traced_model)

class SplitModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y1 = torch.split(x, 3, dim=3)
        y2 = torch.split(x, 4, dim=3)
        y3 = torch.split(x, [2,3,5], dim=2)
        return y1, y2, y3
    
dummy_input = torch.randn((1, 4, 10, 9))
model = SplitModule()
model.eval()
traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("split.pth", traced_model)

class SplitStackModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = torch.split(x, 2, dim=1)
        y = torch.stack(y, 2)
        return y
    
dummy_input = torch.randn((1, 8, 2, 3))
model = SplitStackModule()
model.eval()
traced_model = torch.jit.trace(model, dummy_input)

TracedModelFactory("split_stack.pth", traced_model)

class UnsqueezeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        w = torch.Tensor([[1.2],[3.4],[4.1]])
        w = w.unsqueeze(1)
        return x + w
    
x = torch.randn(1,3)
model = UnsqueezeModule()
model.eval()
traced_model = torch.jit.trace(model, (x))

TracedModelFactory("unsqueeze.pth", traced_model)

class UpsamplingBilinear2dModule(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
#         self.upsamplingBilinear2d = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.upsamplingBilinear2d = torch.nn.UpsamplingBilinear2d(size=(4, 4))
    
    def forward(self, x):
        #res1 = self.upsamplingBilinear2d(x)
        res1 = F.interpolate(x, size=(40,40), mode='bilinear', align_corners=True)
        res2 = F.interpolate(x, size=(40,40), mode='bilinear', align_corners=False)
        return res1, res2        

a = torch.randn(1, 128, 20, 20)
model = UpsamplingBilinear2dModule(2)
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("upsampling_bilinear_2d_with_size.pth", traced_model)

class UpsamplingBilinear2dModule(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.upsamplingBilinear2d = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    
    def forward(self, x):
        # return self.upsamplingBilinear2d(x)
        res1 = F.interpolate(x, scale_factor=(2.0,2.0), mode='bilinear', align_corners=True)
        res2 = F.interpolate(x, scale_factor=(2.0,2.0), mode='bilinear', align_corners=False)
        return res1, res2
    
a = torch.randn(1, 128, 20, 20)
model = UpsamplingBilinear2dModule(2)
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("upsampling_bilinear_2d_with_scale.pth", traced_model)

class UpsamplingNearest2dModule(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.upsamplingNearest2d = torch.nn.UpsamplingNearest2d(scale_factor=scale_factor)
    
    def forward(self, x):
        return self.upsamplingNearest2d(x)
        
    
a = torch.randn(1, 1, 2, 2)
model = UpsamplingNearest2dModule(2.)
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("upsampling_nearest_2d_with_scale.pth", traced_model)

class UpsamplingNearest2dModule(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
#         self.upsamplingNearest2d = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.upsamplingNearest2d = torch.nn.UpsamplingNearest2d(size=(4,4))

    
    def forward(self, x):
        return self.upsamplingNearest2d(x)
        
    
a = torch.randn(1, 1, 2, 2)
model = UpsamplingNearest2dModule(2.)
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("upsampling_nearest_2d_with_size.pth", traced_model)

class VarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y1 = torch.var(x, (1,2,3), True, False)
        y2 = torch.var(x, (2,3), False, True)
        return y1, y2
    
a = torch.randn((3,13,41,39))
model = VarModule()
model.eval()
traced_model = torch.jit.trace(model, a)

TracedModelFactory("var.pth", traced_model)

class ViewModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y = torch.randn((3,5,4))
    
    def forward(self, x):
        d = self.y.size()
        r1 = x.view(d)
        r2 = x.reshape(d)
        return r1, r2
    
dummy_input1 = torch.randn((3, 4, 5))
model = ViewModule()
model.eval()
traced_model = torch.jit.trace(model, dummy_input1)

TracedModelFactory("view_and_reshape.pth", traced_model)

# import torchvision.models as models
# res18 = models.resnet18(pretrained=True)
# res18 = res18.eval().cpu()
# traced_model = torch.jit.trace(res18, torch.randn(1,3,224,224))
# TracedModelFactory('resnet18.pth', traced_model)

# from transformers import BertModel, BertConfig, BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
# model = model.cpu()
# model = model.eval()

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# traced_model = torch.jit.trace(model, tuple(inputs.values()))

# TracedModelFactory("bert.pth", traced_model)

