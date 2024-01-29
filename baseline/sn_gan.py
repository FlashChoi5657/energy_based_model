import torch

from torch import nn
import torch.nn.functional as F

# from sn import SpectralNorm
import numpy as np
from torch.nn.utils import *

class Swish(nn.Module):
    def forward(self: 'Swish', x: torch.Tensor) -> torch.Tensor:
        return (x * torch.sigmoid(x))/1.1

# leak = 0.1
# class Discriminator(nn.Module):
#     def __init__(self, in_C):
#         super(Discriminator, self).__init__()

#         self.conv1 = SpectralNorm(nn.Conv2d(in_C, 64, 3, stride=1, padding=(1,1)))

#         self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
#         self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
#         self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
#         self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
#         self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
#         self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


#         self.fc = SpectralNorm(nn.Linear(4 * 4 * 512, 1))

#     def forward(self, x):
#         m = x
#         m = nn.LeakyReLU(leak)(self.conv1(m))
#         m = nn.LeakyReLU(leak)(self.conv2(m))
#         m = nn.LeakyReLU(leak)(self.conv3(m))
#         m = nn.LeakyReLU(leak)(self.conv4(m))
#         m = nn.LeakyReLU(leak)(self.conv5(m))
#         m = nn.LeakyReLU(leak)(self.conv6(m))
#         m = nn.LeakyReLU(leak)(self.conv7(m))

#         return self.fc(m.view(-1, 4 * 4 * 512))
    

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, activation='silu', stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'swish':
            self.activation = Swish()
        elif activation == 'elu':
            self.activation = nn.ELU()
        
        self.model = nn.Sequential(
            self.activation,
            spectral_norm(self.conv1),
            self.activation,
            spectral_norm(self.conv2)
            )

        self.bypass = nn.Sequential()
        if in_channels != out_channels or stride != 1:
        
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=False)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )
        
        self.down = None
        if stride != 1:
            self.down = nn.Sequential(
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        out = self.model(x) + self.bypass(x)
        if self.down is not None:
            out = self.down(out)
        return out

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, activation='silu'):
        super(FirstResBlockDiscriminator, self).__init__()
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'swish':
            self.activation = Swish()
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            self.activation,
            spectral_norm(self.conv2),
            )
        self.bypass = nn.Sequential(
        )

    def forward(self, x):
        return self.model(x) 


class Discriminator(nn.Module):
    def __init__(self, in_C, dim_feature, activation='silu'):
        super(Discriminator, self).__init__()
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'swish':
            self.activation = Swish()
        
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(in_C, dim_feature, activation=activation),
                ResBlockDiscriminator(dim_feature, dim_feature, activation=activation, stride=2),
                ResBlockDiscriminator(dim_feature, dim_feature * 2, activation=activation),
                ResBlockDiscriminator(dim_feature *2 ,dim_feature *2, activation=activation, stride=2),
                ResBlockDiscriminator(dim_feature *2, dim_feature *2, activation=activation),
                ResBlockDiscriminator(dim_feature *2, dim_feature *4, activation=activation, stride=2),
                ResBlockDiscriminator(dim_feature *4, dim_feature *4, activation=activation),
                ResBlockDiscriminator(dim_feature *4, dim_feature *4, activation=activation, stride=2),
                ResBlockDiscriminator(dim_feature *4, dim_feature *8, activation=activation),
                self.activation,
            )
        self.fc = nn.Linear(dim_feature * 8, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = spectral_norm(self.fc)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.fc(out)
        return out