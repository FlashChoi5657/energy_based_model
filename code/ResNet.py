from torch import nn
from torch.autograd import grad
import torch

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn import utils


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


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        # nn.init.normal_(module.weight, 0, std)
        nn.init.xavier_uniform_(module.weight.data)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation, downsample=False):
        super().__init__()

        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False,
            )
        )

        self.conv2 = spectral_norm(
            nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False,
            ), std=1e-10, bound=True
        )

        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
            )

        self.downsample = downsample
        if activation == 'silu':
            self.activation = Swish()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
            
            
    def forward(self, input):
        out = input

        out = self.conv1(out)
        out = self.activation(out)
        # out = F.leaky_relu(out, negative_slope=0.2)

        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)
            
        out = self.activation(out)

        # out = F.leaky_relu(out, negative_slope=0.2)

        return out


class IGEBM(nn.Module):
    def __init__(self: 'IGEBM', in_channel, dim_feature, out_channel, activation='silu', n_class=None, use_gp: bool = False, gp_weight: float = 0.0, 
                            use_reg: bool = False, reg_weight: float = 0.0, 
                            use_L2_reg: bool = False, L2_reg_weight: float = 0.0, 
                            scale_range: list = [-1, 1]): 
        super(IGEBM, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, dim_feature, 3, padding=1)
        self.conv2 = nn.Conv2d(dim_feature, dim_feature, 3, padding=1)
        self.activation = Swish()
        
        self.firstblock = nn.Sequential(
            spectral_norm(self.conv1, std=1),
            self.activation,
            spectral_norm(self.conv2, std=1)
        )
        self.use_gp        = use_gp
        self.gp_weight     = gp_weight
        self.use_reg       = use_reg
        self.reg_weight    = reg_weight
        self.use_L2_reg    = use_L2_reg
        self.L2_reg_weight = L2_reg_weight
        self.scale_range   = scale_range

        self.blocks = nn.ModuleList(
            [
                ResBlock(dim_feature, dim_feature, activation, downsample=False),
                ResBlock(dim_feature, dim_feature * 2, activation, downsample=True),
                ResBlock(dim_feature * 2, dim_feature * 2, activation, downsample=False),
                ResBlock(dim_feature * 2, dim_feature * 4, activation, downsample=True),
                ResBlock(dim_feature * 4, dim_feature * 4, activation, downsample=False),
                ResBlock(dim_feature * 4, dim_feature * 8, activation, downsample=True),
                # ResBlock(dim_feature * 8, dim_feature * 8, activation, n_class),
            ]
        )
        self.linear = nn.Linear(dim_feature * 8, out_channel)

    def forward(self, input):
        out = self.firstblock(input)
        # out = self.conv1(input)
        # out = self.activation(out)
        # out = F.leaky_relu(out, negative_slope=0.2)

        # out = self.conv2(out)
        # out = self.activation(out)
        
        for block in self.blocks:
            out = block(out)

        # out = F.relu(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out

    def compute_gradient_norm(self: 'IGEBM', x_input: torch.Tensor, x_prediction: torch.Tensor) -> torch.Tensor:
        grad_output = torch.autograd.Variable(torch.ones_like(x_prediction), requires_grad=False)
        gradient    = torch.autograd.grad(outputs=x_prediction, 
                                          inputs=x_input, 
                                          grad_outputs=grad_output, 
                                          create_graph=True, 
                                          retain_graph=True, 
                                          only_inputs=True)[0]

        return torch.sum(gradient.pow(2).reshape(len(x_input), -1).sum(1))

    def compute_gradient(self: 'IGEBM', input: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gradient = torch.autograd.grad(outputs=prediction.sum(), inputs=input, create_graph=True, retain_graph=True)[0]
        
        return gradient


    def compute_gradient_penalty(self: 'IGEBM', input: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gradient        = self.compute_gradient(input, prediction)
        gradient        = gradient.view(input.shape[0], -1)
        gradient_norm   = torch.norm(gradient, p=2, dim=1)
        penalty         = torch.mean((gradient_norm - 1.0).pow(2))
        
        return penalty



    def compute_loss(self: 'IGEBM', positive_output: torch.Tensor, negative_output: torch.Tensor) -> torch.Tensor:
        positive = self(positive_output)
        negative = self(negative_output)
        loss     = negative.sum() - positive.sum()

        if self.use_L2_reg:
            loss += self.L2_reg_weight * (positive**2 + negative**2).sum()

        if self.use_reg:
            positive_grad_norm = self.compute_gradient_norm(positive_output, positive)
            negative_grad_norm = self.compute_gradient_norm(negative_output, negative)
            loss += self.reg_weight * (positive_grad_norm + negative_grad_norm)

        if self.use_gp:
            # loss += self.gp_weight * self.compute_gradient_penalty(positive_output, negative_output)
            size_batch  = positive_output.shape[0] 
            # alpha = torch.rand(size_batch, 1, 1, 1, device=positive_output.device)
            # interpolate_output  = alpha * positive_output.data + (1 - alpha) * negative_output.data
            
            alpha       = torch.rand(size_batch, 1, 1, 1)
            alpha       = alpha.expand_as(positive_output).to(positive_output.device)
            interpolate_output  = alpha * positive_output.data + (1 - alpha) * negative_output.data
            interpolate_output.requires_grad = True
            # interpolate_output  = torch.nn.parameter(interpolate_output, requires_grad=True)
            interpolate_pred    = self(interpolate_output)
            gradient_penalty    = self.compute_gradient_penalty(interpolate_output, interpolate_pred)
            loss += gradient_penalty
            
        return loss
  
    def compute_loss_inference(self: 'IGEBM', negative_output: torch.Tensor) -> torch.Tensor:
        negative = self(negative_output)
        loss     = - negative.sum()

        return loss
  
    def update_prediction_langevin(self: 'IGEBM', prediction: torch.Tensor, number_step_langevin: int, 
                                                        lr_langevin: float, regular_data: float, add_noise: bool, 
                                                        noise_decay: float = 1.0) -> torch.Tensor:
        update = nn.Parameter(prediction, requires_grad=True)

        for _ in range(number_step_langevin):
            loss = self.compute_loss_inference(update)
            loss.backward()
            
            update.data = update.data - 0.5 * lr_langevin * update.grad

            if regular_data > 0.0:
                update.data = update.data - 0.5 * regular_data * update.data

            if add_noise == True:
                update.data = update.data + 0.5 * noise_decay * np.sqrt(lr_langevin) * torch.randn_like(update)
            
            # activation
            update.data = torch.clamp(update.data, *self.scale_range)
            update.grad.detach_()
            update.grad.zero_()

        return update
    

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)
        
        
    def forward(self, input):
        output = self.conv(input)
        return output
    
class Square(nn.Module):
    def __init__(self):
        super(Square,self).__init__()
        pass
    
    def forward(self,in_vect):
        return in_vect**2
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        pass
    
    def forward(self,in_vect):
        return in_vect*nn.functional.sigmoid(in_vect)
    
class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output
    
class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output
    


# special ResBlock just for the first layer of the discriminator
class FirstResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='silu'):
        super(FirstResBlock, self).__init__()
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'swish':
            self.activation = Swish()
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            self.activation
            )

    def forward(self, x):
        return self.model(x) 
    
class EnResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='silu', avg=False):
        super(EnResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        nn.init.xavier_normal_(self.conv1.weight.data, 1.)
        nn.init.xavier_normal_(self.conv2.weight.data, 1.)
        
        if activation == 'silu':
            self.activation = nn.SiLU()
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
        
        if in_channels != out_channels:
        
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=False)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.bypass_conv
                )
        
        self.down = None
        if avg == True:
            self.down = nn.Sequential(
                # nn.AvgPool2d(2, stride=stride, padding=0)
                nn.AvgPool2d(2)
            )

    def forward(self, x):
        out = self.model(x) + self.bypass(x)
        if self.down is not None:
            out = self.down(out)
        out = self.activation(out)
        return out
        
class EnResnet(nn.Module):
    def __init__(self, in_C: int=3, feature: int=32, activation='silu'):
        super(EnResnet, self).__init__()
        
        self.model = nn.Sequential(
            FirstResBlock(in_C, feature, activation=activation),
            EnResBlock(feature,     feature, activation=activation),
            EnResBlock(feature,     feature * 2, activation=activation),
            EnResBlock(feature * 2, feature * 2, activation=activation),
            EnResBlock(feature * 2, feature * 4, activation=activation),
            EnResBlock(feature * 4, feature * 4, activation=activation),
            EnResBlock(feature * 4, feature * 8, activation=activation, avg=True)
            )
        
        self.fc = nn.Linear(feature * 8, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.fc(out)
        
        return out


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hw, resample=None, normalize=False,AF=nn.ELU()):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.normalize = normalize
        self.bn1 = None
        self.bn2 = None
        self.relu1 = AF
        self.relu2 = AF
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'none':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'none':
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
            
            
    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
                shortcut = input
        else:
            shortcut = self.conv_shortcut(input)
        
        if self.normalize == False:
            output = input
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.relu2(output)
            output = self.conv_2(output)
        else:
            output = input
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.conv_2(output)

        return shortcut + output
    
class Res18_Quadratic(nn.Module):
    #Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res18_Quadratic, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rbc11 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc22 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
        

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output

class Res18_Linear(nn.Module):
    #Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self, inchan, dim, hw, normalize=False,AF=None):
        super(Res18_Linear, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rbc11 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc22 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.nl = AF
        self.ln = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        
        

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.nl(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln(output)
        return output	
	
	
class Res12_Quadratic(nn.Module):
    #6 block resnet used in MIT EBM papaer for conditional Imagenet 32X32
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res12_Quadratic, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
        
        
    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output        

class Res6_Quadratic(nn.Module):
    #3 block resnet for small MNIST experiment
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res6_Quadratic, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)        
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
        
        
    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)        
        output = self.rb2(output)       
        output = self.rb3(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output    
    
    
class Res34_Quadratic(nn.Module):
    #Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res34_Quadratic, self).__init__()
        #made first layer 2*dim wide to not provide bottleneck
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rbc11 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc22 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc222 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc333 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc3333 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33333 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rb4 = ResidualBlock(8*dim, 16*dim, 3, int(hw/8), resample = 'down',normalize=normalize,AF=AF)
        self.rbc4 = ResidualBlock(16*dim, 16*dim, 3, int(hw/16), resample = 'none',normalize=normalize,AF=AF)
        self.rbc44 = ResidualBlock(16*dim, 16*dim, 3, int(hw/16), resample = 'none',normalize=normalize,AF=AF)
        
        
        self.ln1 = nn.Linear(int(hw/16)*int(hw/16)*16*dim, 1)
        self.ln2 = nn.Linear(int(hw/16)*int(hw/16)*16*dim, 1)
        self.lq = nn.Linear(int(hw/16)*int(hw/16)*16*dim, 1)
        self.Square = Square()
        

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rbc222(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.rbc333(output)
        output = self.rbc3333(output)
        output = self.rbc33333(output)
        output = self.rb4(output)
        output = self.rbc4(output)
        output = self.rbc44(output)
        output = output.view(-1, int(self.hw/16)*int(self.hw/16)*16*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output    
    
    
    
    
    
    
    
    