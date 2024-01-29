from torch import nn
from torch.autograd import grad
import torch

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import *

class Swish(nn.Module):
    def forward(self: 'Swish', x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class EnResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, activation='swish', avg=False):
        super(EnResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_normal_(self.conv1.weight.data, 1.)
        nn.init.xavier_normal_(self.conv2.weight.data, 1.)
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'swish':
            self.activation = Swish()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            )
        
        self.bypass = nn.Sequential()
        
        if in_channels != out_channels or stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(self.bypass_conv)
        
        self.down = None
        
        if avg == True:
            self.down = nn.Sequential(nn.AvgPool2d(2))

    def forward(self, x):
        out = self.model(x) + self.bypass(x)
        if self.down is not None:
            out = self.down(out)
        out = self.activation(out)
        return out
        
class EnResnet(nn.Module):
    def __init__(self, in_C: int=3, feature: int=32, out_C: int=1, activation='swish'):
        super(EnResnet, self).__init__()
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2) # 0.05
        elif activation == 'swish':
            self.activation = Swish()
            
        self.conv1 = nn.Conv2d(in_C, feature, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(feature, feature, 3, 1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.firstlayer = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            self.activation
            )
        
        self.model = nn.Sequential(
            EnResBlock(feature,     feature,     activation=activation),
            EnResBlock(feature,     feature * 2, activation=activation),
            EnResBlock(feature * 2, feature * 2, activation=activation),
            EnResBlock(feature * 2, feature * 2, activation=activation, stride=2),
            EnResBlock(feature * 2, feature * 4, activation=activation),
            EnResBlock(feature * 4, feature * 4, activation=activation),
            EnResBlock(feature * 4, feature * 4, activation=activation, stride=2),
            # EnResBlock(feature * 4, feature * 8, activation=activation),
            EnResBlock(feature * 4, feature * 8, activation=activation, avg=True),
            )
        
        self.fc = nn.Linear(feature * 8, out_C)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        
    def forward(self, x):
        out = self.firstlayer(x)
        out = self.model(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.fc(out)
        
        return out


class EnResBlockSN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, activation='swish', avg=False):
        super(EnResBlockSN, self).__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        nn.init.xavier_normal_(self.conv1.weight.data, 1.)
        nn.init.xavier_normal_(self.conv2.weight.data, 1.)
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'swish':
            self.activation = Swish()
        elif activation == 'elu':
            self.activation = nn.ELU()
        
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            )
        
        self.bypass = nn.Sequential()
        
        if in_channels != out_channels or stride != 1:
            self.bypass_conv = spectral_norm(nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(self.bypass_conv)
        
        self.down = None
        
        if avg == True:
            self.down = nn.Sequential(nn.AvgPool2d(2))

    def forward(self, x):
        out = self.model(x) + self.bypass(x)
        if self.down is not None:
            out = self.down(out)
        out = self.activation(out)
        return out
        
class EnResnetSN(nn.Module):
    def __init__(self, in_C: int=3, feature: int=32, activation='lrelu'):
        super(EnResnetSN, self).__init__()
        
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'swish':
            self.activation = Swish()
            
        self.conv1 = nn.Conv2d(in_C, feature, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(feature, feature, 3, 1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.firstlayer = nn.Sequential(
            spectral_norm(self.conv1),
            self.activation,
            spectral_norm(self.conv2),
            self.activation
            )
        
        self.model = nn.ModuleList([
            # self.firstlayer,
            EnResBlockSN(feature,     feature,     activation=activation),
            EnResBlockSN(feature,     feature * 2, activation=activation),
            EnResBlockSN(feature * 2, feature * 2, activation=activation),
            EnResBlockSN(feature * 2, feature * 2, activation=activation, stride=2),
            EnResBlockSN(feature * 2, feature * 4, activation=activation),
            EnResBlockSN(feature * 4, feature * 4, activation=activation),
            EnResBlockSN(feature * 4, feature * 4, activation=activation, stride=2),
            # EnResBlockSN(feature * 4, feature * 8, activation=activation),
            EnResBlockSN(feature * 4, feature * 8, activation=activation, avg=True),
            ])
        
        self.fc = nn.Linear(feature * 8, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = spectral_norm(self.fc)
        
    def forward(self, x):
        out = self.firstlayer(x)
        for layer in self.model:
            out = layer(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.fc(out)
        
        return out
    
class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_n(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_class=None, downsample=False):
        super().__init__()

        self.conv1 = spectral_n(
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )
        )

        self.conv2 = spectral_n(
            nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            ), std=1e-10, bound=True
        )

        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                spectral_n(nn.Conv2d(in_channel, out_channel, 1, bias=False))
            )

        self.downsample = downsample

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = F.leaky_relu(out, negative_slope=0.2)

        out = self.conv2(out)

        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = F.leaky_relu(out, negative_slope=0.2)

        return out


class IGEBM(nn.Module):
    def __init__(self, n_class=None):
        super().__init__()

        self.conv1 = spectral_n(nn.Conv2d(6, 128, 3, padding=1), std=1)

        self.blocks = nn.ModuleList(
            [
                ResBlock(128, 128, n_class, downsample=True),
                ResBlock(128, 128, n_class),
                ResBlock(128, 256, n_class, downsample=True),
                ResBlock(256, 256, n_class),
                ResBlock(256, 256, n_class, downsample=True),
                ResBlock(256, 256, n_class),
            ]
        )

        self.linear = nn.Linear(256, 1)

    def forward(self, a, b, class_id=None):
        input = torch.cat([a,b], dim=1)
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out
    