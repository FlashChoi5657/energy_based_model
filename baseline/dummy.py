from torch import nn
from torch.autograd import grad
import torch

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F


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
