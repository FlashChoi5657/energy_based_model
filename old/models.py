import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from unet_sub_models import DoubleConv, Down, Up, OutConv


# ======================================================================
# UNet
# ======================================================================
class UNet(nn.Module):
    def __init__(self, in_channel: int, dim_feature: int, out_channel: int, bilinear: bool = False, out_activation: str = 'tanh') -> None:
        super(UNet, self).__init__()
        
        self.in_channel     = in_channel
        self.dim_feature    = dim_feature
        self.out_channel    = out_channel
        self.bilinear       = bilinear
        self.out_activation = out_activation
        self.criterion      = nn.MSELoss(reduction='sum')

        self.inc   = DoubleConv(in_channel, dim_feature * 2)
        self.down1 = Down(dim_feature * 2, dim_feature * 4)
        self.down2 = Down(dim_feature * 4, dim_feature * 8)
        self.down3 = Down(dim_feature * 8, dim_feature * 16)
        
        factor     = 2 if bilinear else 1
        self.down4 = Down(dim_feature * 16, dim_feature * 32 // factor)
        self.up1   = Up(dim_feature * 32, dim_feature * 16 // factor, bilinear)
        self.up2   = Up(dim_feature * 16, dim_feature * 8 // factor, bilinear)
        self.up3   = Up(dim_feature * 8, dim_feature * 4 // factor, bilinear)
        self.up4   = Up(dim_feature * 4, dim_feature * 2, bilinear)
        self.outc  = OutConv(dim_feature * 2, out_channel)

        if self.out_activation == 'tanh':
            self.output = nn.Tanh()

        elif self.out_activation == 'sigmoid':
            self.output = nn.Sigmoid()

    def forward(self: 'UNet', x: torch.Tensor) -> torch.Tensor:
        x1     = self.inc(x)
        x2     = self.down1(x1)
        x3     = self.down2(x2)
        x4     = self.down3(x3)
        x5     = self.down4(x4)
        x      = self.up1(x5, x4)
        x      = self.up2(x, x3)
        x      = self.up3(x, x2)
        x      = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.output(logits)

        return logits

    def compute_loss(self: 'UNet', prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_recon = self.criterion(prediction, target)

        return loss_recon


# ======================================================================
# Encoder
# ======================================================================
class Encoder(nn.Module):
    def __init__(self: 'Encoder', in_channel: int = 1, dim_feature: int = 32, dim_latent: int = 128) -> None:
        super(Encoder, self).__init__()

        self.in_channel  = in_channel
        self.dim_feature = dim_feature 
        self.dim_latent  = dim_latent

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=dim_feature * 1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim_feature * 1, out_channels=dim_feature * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dim_feature * 2, out_channels=dim_feature * 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=dim_feature * 4, out_channels=dim_feature * 8, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=dim_feature * 8, out_channels=dim_feature * 16, kernel_size=3, stride=2, padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(dim_feature * 1)
        self.bn2 = nn.BatchNorm2d(dim_feature * 2)
        self.bn3 = nn.BatchNorm2d(dim_feature * 4)
        self.bn4 = nn.BatchNorm2d(dim_feature * 8)
        self.bn5 = nn.BatchNorm2d(dim_feature * 16)

        self.linear_mean = nn.Linear(in_features=dim_feature * 16, out_features=dim_latent, bias=True)
        self.linear_var  = nn.Linear(in_features=dim_feature * 16, out_features=dim_latent, bias=True)

        self.activation = nn.LeakyReLU()
        self.flatten    = nn.Flatten()

    def forward(self: 'Encoder', x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.flatten(x)

        mu      = self.linear_mean(x)
        log_var = self.linear_var(x)

        return mu, log_var


# ======================================================================
# Decoder
# ======================================================================
class Decoder(nn.Module):
    def __init__(self: 'Decoder', dim_latent: int = 128, dim_feature: int = 32, out_channel: int = 1, out_activation: str = 'tanh') -> None:
        super(Decoder, self).__init__()

        self.dim_latent     = dim_latent
        self.dim_feature    = dim_feature
        self.out_channel    = out_channel 
        self.out_activation = out_activation

        self.input     = nn.Linear(dim_latent, dim_feature * 16)
        self.unflatten = nn.Unflatten(1, (dim_feature * 16, 1, 1))
        self.upsample  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(in_channels=dim_feature * 16, out_channels=dim_feature * 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim_feature * 8, out_channels=dim_feature * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dim_feature * 4, out_channels=dim_feature * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=dim_feature * 2, out_channels=dim_feature * 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=dim_feature * 1, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(dim_feature * 8)
        self.bn2 = nn.BatchNorm2d(dim_feature * 4)
        self.bn3 = nn.BatchNorm2d(dim_feature * 2)
        self.bn4 = nn.BatchNorm2d(dim_feature * 1)

        self.activation = nn.LeakyReLU()
        if self.out_activation == 'tanh':
            self.output = nn.Tanh()

        elif self.out_activation == 'sigmoid':
            self.output = nn.Sigmoid()
      
    def forward(self: 'Decoder', x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.output(x)

        return x


# ======================================================================
# VAE
# ======================================================================
class VAE(nn.Module):
    def __init__(self: 'VAE', in_channel: int = 1, dim_feature: int = 32, dim_latent: int = 128, out_activation: str = 'tanh', kld_weight: float = 1.0) -> None:
        super(VAE, self).__init__()

        self.encoder    = Encoder(in_channel, dim_feature, dim_latent)
        self.decoder    = Decoder(dim_latent, dim_feature, in_channel, out_activation)
        self.dim_latent = self.encoder.dim_latent
        self.criterion  = nn.MSELoss(reduction='sum')
        self.kld_weight  = kld_weight
      
    def reparameterization(self: 'VAE', mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_var)
        z     = mu + sigma * torch.randn_like(sigma)

        return z

    def compute_latent(self: 'VAE', x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z           = self.reparameterization(mu, log_var)

        return z, mu, log_var

    def forward(self: 'VAE', x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, log_var = self.compute_latent(x)
        y              = self.decoder(z)

        return y, mu, log_var, z

    def compute_loss(self: 'VAE', prediction: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_recon = self.criterion(prediction, target)
        loss_kld   = -0.5 * torch.sum(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        loss       = loss_recon + self.kld_weight * loss_kld        

        return loss, loss_recon, loss_kld
  
    def sample(self: 'VAE', number_sample: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(number_sample, self.dim_latent).to(device)
        y = self.decoder(z)

        return y


# ======================================================================
# Energy Model
# ======================================================================

class Swish(nn.Module):
    def forward(self: 'Swish', x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    def __init__(self: 'EnergyModel', in_channel: int = 1, dim_feature: int = 32, dim_output: int = 1, dim_latent: int = 0, 
                                                                                                       use_gp: bool = False, gp_weight: float = 0.0, 
                                                                                                       use_reg: bool = False, reg_weight: float = 0.0, 
                                                                                                       use_L2_reg: bool = False, L2_reg_weight: float = 0.0, 
                                                                                                       scale_range: list = [-1, 1]) -> None:
        super(EnergyModel, self).__init__()

        self.in_channel    = in_channel
        self.dim_feature   = dim_feature
        self.dim_output    = dim_output
        self.dim_latent    = dim_latent
        self.use_gp        = use_gp
        self.gp_weight     = gp_weight
        self.use_reg       = use_reg
        self.reg_weight    = reg_weight
        self.use_L2_reg    = use_L2_reg
        self.L2_reg_weight = L2_reg_weight
        self.scale_range   = scale_range

        self.dim_latent_z  = dim_latent
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=dim_feature * 1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim_feature * 1, out_channels=dim_feature * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dim_feature * 2, out_channels=dim_feature * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.linear1 = nn.Linear(in_features=dim_latent, out_features=dim_feature * 1, bias=True)
        self.linear2 = nn.Linear(in_features=dim_latent, out_features=dim_feature * 2, bias=True)
        self.linear3 = nn.Linear(in_features=dim_latent, out_features=dim_feature * 4, bias=True)

        self.linear            = nn.Linear(in_features=dim_feature * 4 * 4 * 4 + dim_latent, out_features=dim_output, bias=True)
        self.activation: Swish = Swish()
        self.flatten           = nn.Flatten()

    def forward(self: 'EnergyModel', x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1(x)
        
        if z is not None:
            z = self.linear1(z)
            z = z.unsqueeze(-1) 
            z = z.unsqueeze(-1) 
            x = self.activation(x + z)
        else: 
            x = self.activation(x)

        x = self.conv2(x)

        if z is not None:
            z = self.linear2(z)
            z = z.unsqueeze(-1) 
            z = z.unsqueeze(-1) 
            x = self.activation(x + z)
        else: 
            x = self.activation(x)
  
        x = self.conv3(x)
        
        if z is not None:
            z = self.linear3(z)
            z = z.unsqueeze(-1) 
            z = z.unsqueeze(-1) 
            x = self.activation(x + z)
        else: 
            x = self.activation(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x

    def compute_gradient_norm(self: 'EnergyModel', x_input: torch.Tensor, x_prediction: torch.Tensor) -> torch.Tensor:
        grad_output = torch.autograd.Variable(torch.ones_like(x_prediction), requires_grad=False)
        gradient    = torch.autograd.grad(outputs=x_prediction, 
                                          inputs=x_input, 
                                          grad_outputs=grad_output, 
                                          create_graph=True, 
                                          retain_graph=True, 
                                          only_inputs=True)[0]

        return torch.sum(gradient.pow(2).reshape(len(x_input), -1).sum(1))
    '''
    def compute_gradient_penalty(self: 'EnergyModel', positive_output: torch.Tensor, negative_output: torch.Tensor) -> torch.Tensor:
        batch_size = negative_output.shape[0]
        alpha      = torch.rand((batch_size, 1, 1, 1), device=positive_output.device)
        
        interpolate = (alpha * positive_output) + (1.0 - alpha) * negative_output
        interpolate = interpolate.requires_grad_(True)
        
        prediction  = self(interpolate)
        grad_output = torch.autograd.Variable(torch.ones_like(prediction), requires_grad=False)
        
        gradient = torch.autograd.grad(outputs=prediction, 
                                       inputs=interpolate, 
                                       grad_outputs=grad_output, 
                                       create_graph=True, 
                                       retain_graph=True, 
                                       only_inputs=True)[0]
        penalty = torch.sum((gradient.view(batch_size, -1).norm(2, 1) - 1).pow(2))
        
        return penalty
    '''


    def compute_gradient(self: 'CondEnergyModel', input: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gradient = torch.autograd.grad(outputs=prediction.sum(), inputs=input, create_graph=True, retain_graph=True)[0]
        
        return gradient


    def compute_gradient_penalty(self: 'CondEnergyModel', input: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gradient        = self.compute_gradient(input, prediction)
        gradient        = gradient.view(input.shape[0], -1)
        gradient_norm   = torch.norm(gradient, p=2, dim=1)
        penalty         = torch.mean((gradient_norm - 1.0).pow(2))
        
        return penalty



    def compute_loss(self: 'EnergyModel', positive_output: torch.Tensor, negative_output: torch.Tensor, z_negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        positive = self(positive_output, z_negative)
        negative = self(negative_output, z_negative)
        loss     = positive.sum() - negative.sum()

        if self.use_L2_reg:
            loss += self.L2_reg_weight * (positive**2 + negative**2).sum()

        if self.use_reg:
            positive_grad_norm = self.compute_gradient_norm(positive_output, positive)
            negative_grad_norm = self.compute_gradient_norm(negative_output, negative)
            loss += self.reg_weight * (positive_grad_norm + negative_grad_norm)

        if self.use_gp:
            # loss += self.gp_weight * self.compute_gradient_penalty(positive_output, negative_output)
            size_batch  = positive_output.shape[0] 
            alpha       = torch.rand(size_batch, 1, 1, 1)
            alpha       = alpha.expand_as(positive_output).to(positive_output.device)
            interpolate_output  = alpha * positive_output.data + (1 - alpha) * negative_output.data
            interpolate_output  = torch.nn.parameter(interpolate_output, requires_grad=True)
            interpolate_pred    = self(interpolate_output)
            gradient_penalty    = self.compute_gradient_penalty(interpolate_output, interpolate_pred)
            loss += gradient_penalty
            
        return loss
  
    def compute_loss_inference(self: 'EnergyModel', negative_output: torch.Tensor, z_negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        negative = self(negative_output, z_negative)
        loss     = -negative.sum()

        return loss
  
    def update_prediction_langevin(self: 'EnergyModel', prediction: torch.Tensor, number_step_langevin: int, 
                                                        lr_langevin: float, regular_data: float, add_noise: bool, 
                                                        noise_decay: float = 1.0, z_negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        update = nn.Parameter(prediction, requires_grad=True)

        for _ in range(number_step_langevin):
            loss = self.compute_loss_inference(update, z_negative)
            loss.backward()
            
            update.data = update.data + 0.5 * lr_langevin * update.grad

            if regular_data > 0.0:
                update.data = update.data - 0.5 * regular_data * update.data

            if add_noise == True:
                update.data = update.data + 0.5 * noise_decay * np.sqrt(lr_langevin) * torch.randn_like(update)
            
            # activation
            update.data = torch.clamp(update.data, *self.scale_range)
            update.grad.detach_()
            update.grad.zero_()

        return update


class CondEnergyModel(nn.Module):
    def __init__(self: 'CondEnergyModel', in_channel: int = 2, dim_feature: int = 32, dim_output: int = 1, dim_latent: int = 0, 
                                                                                                           use_gp: bool = False, gp_weight: float = 0.0, 
                                                                                                           use_reg: bool = False, reg_weight: float = 0.0, 
                                                                                                           use_L2_reg: bool = False, L2_reg_weight: float = 0.0, 
                                                                                                           scale_range: list = [-1, 1]) -> None:
        super(CondEnergyModel, self).__init__()

        self.in_channel    = in_channel
        self.dim_feature   = dim_feature
        self.dim_output    = dim_output
        self.dim_latent    = dim_latent
        self.use_gp        = use_gp
        self.gp_weight     = gp_weight
        self.use_reg       = use_reg
        self.reg_weight    = reg_weight
        self.use_L2_reg    = use_L2_reg
        self.L2_reg_weight = L2_reg_weight
        self.scale_range   = scale_range

        self.dim_latent_z  = dim_latent
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=dim_feature * 1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dim_feature * 1, out_channels=dim_feature * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dim_feature * 2, out_channels=dim_feature * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.linear1 = nn.Linear(in_features=dim_latent, out_features=dim_feature * 1, bias=True)
        self.linear2 = nn.Linear(in_features=dim_latent, out_features=dim_feature * 2, bias=True)
        self.linear3 = nn.Linear(in_features=dim_latent, out_features=dim_feature * 4, bias=True)

        self.linear            = nn.Linear(in_features=dim_feature * 4 * 4 * 4 + dim_latent, out_features=dim_output, bias=True)
        self.activation: Swish = Swish()
        self.flatten           = nn.Flatten()

    def forward(self: 'CondEnergyModel', a: torch.Tensor, b: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)
        x = self.conv1(x)
        
        if z is not None:
            z = self.linear1(z)
            z = z.unsqueeze(-1) 
            z = z.unsqueeze(-1) 
            x = self.activation(x + z)
        else: 
            x = self.activation(x)

        x = self.conv2(x)

        if z is not None:
            z = self.linear2(z)
            z = z.unsqueeze(-1) 
            z = z.unsqueeze(-1) 
            x = self.activation(x + z)
        else: 
            x = self.activation(x)
  
        x = self.conv3(x)
        
        if z is not None:
            z = self.linear3(z)
            z = z.unsqueeze(-1) 
            z = z.unsqueeze(-1) 
            x = self.activation(x + z)
        else: 
            x = self.activation(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x


    def compute_gradient_norm(self: 'CondEnergyModel', x_input: torch.Tensor, x_prediction: torch.Tensor) -> torch.Tensor:
        grad_output = torch.autograd.Variable(torch.ones_like(x_prediction), requires_grad=False)
        gradient    = torch.autograd.grad(outputs=x_prediction, 
                                          inputs=x_input, 
                                          grad_outputs=grad_output, 
                                          create_graph=True, 
                                          retain_graph=True, 
                                          only_inputs=True)[0]

        return torch.sum(gradient.pow(2).reshape(len(x_input), -1).sum(1))

    '''
    def compute_gradient_penalty(self: 'CondEnergyModel', positive_input: torch.Tensor, positive_output: torch.Tensor, 
                                                          negative_input: torch.Tensor, negative_output: torch.Tensor, 
                                                          z_positive: torch.Tensor) -> torch.Tensor:
        batch_size = negative_output.shape[0]
        alpha      = torch.rand((batch_size, 1, 1, 1), device=positive_output.device)
        
        interpolate_input = (alpha * positive_input) + (1.0 - alpha) * negative_input
        interpolate       = (alpha * positive_output) + (1.0 - alpha) * negative_output
        interpolate       = interpolate.requires_grad_(True)
        
        prediction  = self(interpolate_input, interpolate, z_positive)
        grad_output = torch.autograd.Variable(torch.ones_like(prediction), requires_grad=False)
        
        gradient = torch.autograd.grad(outputs=prediction, 
                                       inputs=interpolate, 
                                       grad_outputs=grad_output, 
                                       create_graph=True, 
                                       retain_graph=True, 
                                       only_inputs=True)[0]
        penalty = torch.sum((gradient.view(batch_size, -1).norm(2, 1) - 1).pow(2))
        
        return penalty
    '''


    def compute_gradient(self: 'CondEnergyModel', input: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gradient = torch.autograd.grad(outputs=prediction.sum(), inputs=input, create_graph=True, retain_graph=True)[0]
        
        return gradient


    def compute_gradient_penalty(self: 'CondEnergyModel', input: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        gradient        = self.compute_gradient(input, prediction)
        gradient        = gradient.view(input.shape[0], -1)
        gradient_norm   = torch.norm(gradient, p=2, dim=1)
        penalty         = torch.mean((gradient_norm - 1.0).pow(2))
        
        return penalty


    def compute_loss(self: 'CondEnergyModel', positive_input: torch.Tensor, positive_output: torch.Tensor, 
                                              negative_input: torch.Tensor, negative_output: torch.Tensor, 
                                              z_positive: Optional[torch.Tensor] = None, 
                                              z_negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        positive_output = positive_output.requires_grad_(True)
        positive        = self(positive_input, positive_output, z_positive)
        negative_output = negative_output.requires_grad_(True)
        negative        = self(negative_input, negative_output, z_negative)
        loss            = positive.sum() - negative.sum()

        if self.use_L2_reg:
            loss += self.L2_reg_weight * (positive**2 + negative**2).sum()

        if self.use_reg:
            positive_grad_norm = self.compute_gradient_norm(positive_output, positive)
            negative_grad_norm = self.compute_gradient_norm(negative_output, negative)
            loss += self.reg_weight * (positive_grad_norm + negative_grad_norm)

        if self.use_gp:
            '''
            loss += self.gp_weight * self.compute_gradient_penalty(positive_input, positive_output, 
                                                                   negative_input, negative_output, 
                                                                   z_positive)
            '''
            size_batch  = positive_output.shape[0] 
            alpha       = torch.rand(size_batch, 1, 1, 1)
            alpha       = alpha.expand_as(positive_output).to(positive_output.device)
            interpolate_input   = alpha * positive_input.data + (1 - alpha) * negative_input.data
            interpolate_output  = alpha * positive_output.data + (1 - alpha) * negative_output.data
            interpolate_input   = torch.nn.Parameter(interpolate_input, requires_grad=True)
            interpolate_output  = torch.nn.Parameter(interpolate_output, requires_grad=True)
            interpolate_pred    = self(interpolate_input, interpolate_output)
            gradient_penalty    = self.compute_gradient_penalty(interpolate_output, interpolate_pred)
            loss += gradient_penalty

        return loss
  
    def compute_loss_inference(self: 'CondEnergyModel', negative_input: torch.Tensor, negative_output: torch.Tensor, z_negative: Optional[torch.Tensor] = None):
        negative = self(negative_input, negative_output, z_negative)
        loss     = -negative.sum()

        return loss
  
    def update_prediction_langevin(self: 'CondEnergyModel', x: torch.Tensor, prediction: torch.Tensor, 
                                                            number_step_langevin: int, lr_langevin: float, 
                                                            regular_data: float, add_noise: bool, 
                                                            noise_decay: float = 1.0, z_negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        update = nn.Parameter(prediction, requires_grad=True)

        for _ in range(number_step_langevin):
            loss = self.compute_loss_inference(x, update, z_negative)
            loss.backward()
            
            update.data = update.data + 0.5 * lr_langevin * update.grad

            if regular_data > 0.0:
                update.data = update.data - 0.5 * regular_data * update.data

            if add_noise == True:
                update.data = update.data + 0.5 * noise_decay * np.sqrt(lr_langevin) * torch.randn_like(update)
            
            # activation
            update.data = torch.clamp(update.data, *self.scale_range)
            update.grad.detach_()
            update.grad.zero_()

        return update