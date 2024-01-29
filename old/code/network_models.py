import torch
from torch import nn
import numpy as np
import torchvision.models as models
import collections.abc


# ======================================================================
# convolutional block for UNET
# ======================================================================
class Conv_Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, activation: str='leakyrelu', use_batch_norm: bool=True):
        super(Conv_Layer, self).__init__()

        self.conv       = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn         = nn.BatchNorm2d(out_channels)

        self.use_batch_norm = use_batch_norm
        
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


# ======================================================================
# convolutional block for UNET
# ======================================================================
class Conv_Double_Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, activation: str='leakyrelu', use_batch_norm: bool=True):
        super(Conv_Double_Layer, self).__init__()

        self.conv1      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2      = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.bn2        = nn.BatchNorm2d(out_channels)
        
        self.use_batch_norm = use_batch_norm
        
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm: 
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_batch_norm: 
            x = self.bn2(x)
        x = self.activation(x)
        return x


# ======================================================================
# convolutional block for UNET
# ======================================================================
class Conv_Double_Resnet_Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, activation: str='leakyrelu', use_batch_norm: bool=True):
        super(Conv_Double_Resnet_Layer, self).__init__()

        self.conv1      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2      = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.bn2        = nn.BatchNorm2d(out_channels)
        
        self.use_batch_norm = use_batch_norm

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
 
    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm: 
            x = self.bn1(x)
        r = self.activation(x)
        x = self.conv2(x)
        if self.use_batch_norm: 
            x = self.bn2(x)
        x = x + r
        x = self.activation(x)
        return x


# ======================================================================
# encoder 
# ======================================================================
class encoder(nn.Module):
    def __init__(self, 
                dim_channel: int=1, 
                dim_feature: int=32, 
                dim_latent: int=128, 
                model_conv: str='conv_double_resnet', 
                activation: str='leakyrelu', 
                use_batch_norm: bool=True, 
                use_skip: bool=True,
                use_dual_input: bool=False):
        
        super(encoder, self).__init__()

        # ===================================================================== 
        # common layers
        # ===================================================================== 
        self.flatten            = nn.Flatten()
        self.downsample         = nn.MaxPool2d((2, 2))
        self.use_skip           = use_skip
        self.use_dual_input     = use_dual_input
        
        # ===================================================================== 
        # convolution layer
        # ===================================================================== 
        if model_conv == 'conv_double_resnet':
            if use_dual_input:
                self.conv1 = Conv_Double_Resnet_Layer(in_channels=dim_channel*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
            else:
                self.conv1 = Conv_Double_Resnet_Layer(in_channels=dim_channel*1, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
            self.conv2 = Conv_Double_Resnet_Layer(in_channels=dim_feature*1, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
            self.conv3 = Conv_Double_Resnet_Layer(in_channels=dim_feature*2, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
            self.conv4 = Conv_Double_Resnet_Layer(in_channels=dim_feature*4, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
            self.conv5 = Conv_Double_Resnet_Layer(in_channels=dim_feature*8, out_channels=dim_feature*16, activation=activation, use_batch_norm=use_batch_norm)
        elif model_conv == 'conv_double':
            if use_dual_input:
                self.conv1 = Conv_Double_Layer(in_channels=dim_channel*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
            else:
                self.conv1 = Conv_Double_Layer(in_channels=dim_channel*1, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
            self.conv2 = Conv_Double_Layer(in_channels=dim_feature*1, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
            self.conv3 = Conv_Double_Layer(in_channels=dim_feature*2, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
            self.conv4 = Conv_Double_Layer(in_channels=dim_feature*4, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
            self.conv5 = Conv_Double_Layer(in_channels=dim_feature*8, out_channels=dim_feature*16, activation=activation, use_batch_norm=use_batch_norm)
        elif model_conv == 'conv':
            if use_dual_input:
                self.conv1 = Conv_Layer(in_channels=dim_channel*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
            else:
                self.conv1 = Conv_Layer(in_channels=dim_channel*1, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
            self.conv2 = Conv_Layer(in_channels=dim_feature*1, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
            self.conv3 = Conv_Layer(in_channels=dim_feature*2, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
            self.conv4 = Conv_Layer(in_channels=dim_feature*4, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
            self.conv5 = Conv_Layer(in_channels=dim_feature*8, out_channels=dim_feature*16, activation=activation, use_batch_norm=use_batch_norm)
        
        # ===================================================================== 
        # linear layer
        # ===================================================================== 
        self.linear = nn.Linear(in_features=dim_feature*16, out_features=dim_latent, bias=True)
        
    
    def forward(self, x, y=None):
        if self.use_dual_input:
            x = torch.cat([x, y], axis=1)             
        
        x1 = self.conv1(x)
        x = self.downsample(x1)
        
        x2 = self.conv2(x)
        x = self.downsample(x2)
        
        x3 = self.conv3(x)
        x = self.downsample(x3)
        
        x4 = self.conv4(x)
        x = self.downsample(x4)
         
        x5 = self.conv5(x)
        x = self.downsample(x5)
       
        x = self.flatten(x)
        z = self.linear(x)

        if self.use_skip: 
            return z, x1, x2, x3, x4, x5
        else:
            return z 
            

# ======================================================================
# decoder
# ======================================================================
class decoder(nn.Module):
    def __init__(self, 
                dim_channel: int=1, 
                dim_feature: int=32,
                dim_latent: int=128, 
                model_conv: str='conv_double_resnet', 
                activation: str='leakyrelu', 
                activation_output: str='sigmoid', 
                use_batch_norm: bool=True, 
                use_skip: bool=True):
        
        super(decoder, self).__init__()

        # ===================================================================== 
        # common layers
        # ===================================================================== 
        self.unflatten  = nn.Unflatten(1, (dim_feature * 16, 1, 1))
        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.use_skip   = use_skip 
       
        if activation_output == 'sigmoid':
            self.output = nn.Sigmoid()
        elif activation_output == 'tanh':
            self.output = nn.Tanh()
 
        # ===================================================================== 
        # convolution layer
        # ===================================================================== 
        if model_conv == 'conv_double_resnet':
            if use_skip:
                self.conv5 = Conv_Double_Resnet_Layer(in_channels=2*dim_feature*16, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
                self.conv4 = Conv_Double_Resnet_Layer(in_channels=2*dim_feature*8, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
                self.conv3 = Conv_Double_Resnet_Layer(in_channels=2*dim_feature*4, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
                self.conv2 = Conv_Double_Resnet_Layer(in_channels=2*dim_feature*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
                self.conv1 = Conv_Double_Resnet_Layer(in_channels=2*dim_feature*1, out_channels=dim_channel*1, activation=activation, use_batch_norm=use_batch_norm)
            else:
                self.conv5 = Conv_Double_Resnet_Layer(in_channels=dim_feature*16, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
                self.conv4 = Conv_Double_Resnet_Layer(in_channels=dim_feature*8, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
                self.conv3 = Conv_Double_Resnet_Layer(in_channels=dim_feature*4, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
                self.conv2 = Conv_Double_Resnet_Layer(in_channels=dim_feature*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
                self.conv1 = Conv_Double_Resnet_Layer(in_channels=dim_feature*1, out_channels=dim_channel*1, activation=activation, use_batch_norm=use_batch_norm)

        elif model_conv == 'conv_double':
            if use_skip:
                self.conv5 = Conv_Double_Layer(in_channels=2*dim_feature*16, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
                self.conv4 = Conv_Double_Layer(in_channels=2*dim_feature*8, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
                self.conv3 = Conv_Double_Layer(in_channels=2*dim_feature*4, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
                self.conv2 = Conv_Double_Layer(in_channels=2*dim_feature*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
                self.conv1 = Conv_Double_Layer(in_channels=2*dim_feature*1, out_channels=dim_channel*1, activation=activation, use_batch_norm=use_batch_norm)
            else:
                self.conv5 = Conv_Double_Layer(in_channels=dim_feature*16, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
                self.conv4 = Conv_Double_Layer(in_channels=dim_feature*8, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
                self.conv3 = Conv_Double_Layer(in_channels=dim_feature*4, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
                self.conv2 = Conv_Double_Layer(in_channels=dim_feature*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
                self.conv1 = Conv_Double_Layer(in_channels=dim_feature*1, out_channels=dim_channel*1, activation=activation, use_batch_norm=use_batch_norm)
                
        elif model_conv == 'conv':
            if use_skip:
                self.conv5 = Conv_Layer(in_channels=2*dim_feature*16, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
                self.conv4 = Conv_Layer(in_channels=2*dim_feature*8, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
                self.conv3 = Conv_Layer(in_channels=2*dim_feature*4, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
                self.conv2 = Conv_Layer(in_channels=2*dim_feature*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
                self.conv1 = Conv_Layer(in_channels=2*dim_feature*1, out_channels=dim_channel*1, activation=activation, use_batch_norm=use_batch_norm)
            else:
                self.conv5 = Conv_Layer(in_channels=dim_feature*16, out_channels=dim_feature*8, activation=activation, use_batch_norm=use_batch_norm)
                self.conv4 = Conv_Layer(in_channels=dim_feature*8, out_channels=dim_feature*4, activation=activation, use_batch_norm=use_batch_norm)
                self.conv3 = Conv_Layer(in_channels=dim_feature*4, out_channels=dim_feature*2, activation=activation, use_batch_norm=use_batch_norm)
                self.conv2 = Conv_Layer(in_channels=dim_feature*2, out_channels=dim_feature*1, activation=activation, use_batch_norm=use_batch_norm)
                self.conv1 = Conv_Layer(in_channels=dim_feature*1, out_channels=dim_channel*1, activation=activation, use_batch_norm=use_batch_norm)
        
        # ===================================================================== 
        # linear layer
        # ===================================================================== 
        self.linear = nn.Linear(in_features=dim_latent, out_features=dim_feature * 16, bias=True)
        
    def forward(self, x, x1=None, x2=None, x3=None, x4=None, x5=None):
        x = self.linear(x)
        x = self.unflatten(x)
        
        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, x5], axis=1)
        x = self.conv5(x)

        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, x4], axis=1)
        x = self.conv4(x)
        
        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, x3], axis=1)
        x = self.conv3(x)
        
        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, x2], axis=1)
        x = self.conv2(x)
       
        x = self.upsample(x)
        if self.use_skip:
            x = torch.cat([x, x1], axis=1)
        x = self.conv1(x)
        x = self.output(x)
        
        return x 


# ======================================================================
# auto-encoder 
# ======================================================================
class auto_encoder(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32, 
        dim_latent: int=128, 
        model_conv: str='conv_double_resnet', 
        activation: str='leakyrelu', 
        activation_output: str='sigmoid', 
        use_batch_norm: bool=True, 
        use_skip: bool=True,
        use_dual_input: bool=False):
        
        super(auto_encoder, self).__init__()
        
        self.encoder = encoder(
            dim_channel=dim_channel,
            dim_feature=dim_feature,
            dim_latent=dim_latent,
            model_conv=model_conv,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_skip=use_skip,
            use_dual_input=use_dual_input)

        self.decoder = decoder(
            dim_channel=dim_channel,
            dim_feature=dim_feature,
            dim_latent=dim_latent,
            model_conv=model_conv,
            activation=activation,
            activation_output=activation_output,
            use_batch_norm=use_batch_norm,
            use_skip=use_skip)
        
        self.use_skip       = use_skip
        self.use_dual_input = use_dual_input
         
    def forward(self, x, y=None):
        # time must be a list and have the same size as batch
        if self.use_skip:
            if self.use_dual_input:
                z, x1, x2, x3, x4, x5 = self.encoder(x, y)
            else:
                z, x1, x2, x3, x4, x5 = self.encoder(x)
            
            h = self.decoder(z, x1, x2, x3, x4, x5)
        else:
            if self.use_dual_input:
                z = self.encoder(x, y)
            else:
                z = self.encoder(x)
            
            h = self.decoder(z)
        
        return h, z


    def compute_loss(self, prediction, target):
        criterion   = nn.MSELoss(reduction='mean')
        loss        = criterion(prediction, target)
        return loss

 
# ======================================================================
# energy 
# ======================================================================
class energy(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32, 
        dim_latent: int=128, 
        model_conv: str='conv_double_resnet', 
        activation: str='leakyrelu', 
        activation_output: str='identity', 
        use_batch_norm: bool=True, 
        use_skip: bool=False,
        use_dual_input: bool=False):
        
        super(energy, self).__init__()
        
        self.encoder = encoder(
            dim_channel=dim_channel,
            dim_feature=dim_feature,
            dim_latent=dim_latent,
            model_conv=model_conv,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_skip=use_skip,
            use_dual_input=use_dual_input)

        self.use_dual_input = use_dual_input
        
        # ===================================================================== 
        # common layers
        # ===================================================================== 
        self.flatten = nn.Flatten()
       
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
             
        if activation_output == 'sigmoid':
            self.output = nn.Sigmoid()
        elif activation_output == 'tanh':
            self.output = nn.Tanh()
        elif activation_output == 'identity':
            self.output = nn.Identity()

        # ===================================================================== 
        # linear layer
        # ===================================================================== 
        self.linear1 = nn.Linear(in_features=dim_latent, out_features=dim_feature*4, bias=True)
        # self.linear2 = nn.Linear(in_features=dim_feature*8, out_features=dim_feature*4, bias=True)
        self.linear3 = nn.Linear(in_features=dim_feature*4, out_features=dim_feature*2, bias=True)
        # self.linear4 = nn.Linear(in_features=dim_feature*2, out_features=dim_feature*1, bias=True)
        self.linear5 = nn.Linear(in_features=dim_feature*2, out_features=1, bias=True)
          

    def forward(self, x, y=None):
        # time must be a list and have the same size as batch
        if self.use_dual_input:
            z = self.encoder(x, y)
        else:
            z = self.encoder(x)
        
        x = self.linear1(z)
        x = self.activation(x)
        # x = self.linear2(x)
        # x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        # x = self.linear4(x)
        # x = self.activation(x)
        x = self.linear5(x)
        x = self.output(x)
        
        return x
    
    def compute_loss_contrastive_divergence_positive(self, prediction_negative, prediction_positive):
        loss = torch.mean(prediction_negative) - torch.mean(prediction_positive)
        return loss

    def compute_loss_contrastive_divergence_negative(self, prediction_negative):
        loss = - torch.mean(prediction_negative)
        return loss

    def compute_loss_positive(self, prediction_negative, prediction_positive):
        loss = self.compute_loss_contrastive_divergence_positive(prediction_negative, prediction_positive)
        return loss

    def compute_loss_negative(self, prediction_negative):
        loss = self.compute_loss_contrastive_divergence_negative(prediction_negative)
        return loss



# ======================================================================
# energy model 
# ======================================================================
class energy2(nn.Module):

    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        use_batch_norm: bool=True,
        use_dual_input: bool=True):

        super(energy2, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.use_dual_input = use_dual_input
        self.activation     = nn.LeakyReLU()
        self.flatten        = nn.Flatten()
 
        if use_dual_input:
            self.conv1  = nn.Conv2d(in_channels=dim_channel*2, out_channels=dim_feature*1, kernel_size=3, stride=2, padding=1, bias=True)
        else:
            self.conv1  = nn.Conv2d(in_channels=dim_channel*1, out_channels=dim_feature*1, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv2  = nn.Conv2d(in_channels=dim_feature*1, out_channels=dim_feature*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3  = nn.Conv2d(in_channels=dim_feature*2, out_channels=dim_feature*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4  = nn.Conv2d(in_channels=dim_feature*4, out_channels=dim_feature*8, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5  = nn.Conv2d(in_channels=dim_feature*8, out_channels=dim_feature*16, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.bn1    = nn.BatchNorm2d(dim_feature*1)
        self.bn2    = nn.BatchNorm2d(dim_feature*2)
        self.bn3    = nn.BatchNorm2d(dim_feature*4)
        self.bn4    = nn.BatchNorm2d(dim_feature*8)
        self.bn5    = nn.BatchNorm2d(dim_feature*16)

        self.linear1    = nn.Linear(in_features=dim_feature*16, out_features=dim_feature*8, bias=True)
        self.linear2    = nn.Linear(in_features=dim_feature*8, out_features=dim_feature*4, bias=True)
        self.linear3    = nn.Linear(in_features=dim_feature*4, out_features=dim_feature*2, bias=True)
        self.linear4    = nn.Linear(in_features=dim_feature*2, out_features=dim_feature*1, bias=True)
        self.linear5    = nn.Linear(in_features=dim_feature*1, out_features=1, bias=True)

    def forward(self, x, y=None):
        if self.use_dual_input:
            x = torch.cat([x, y], axis=1)
            
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.activation(x)
        
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.activation(x)
        
        x = self.conv5(x)
        if self.use_batch_norm:
            x = self.bn5(x)
        x = self.activation(x)
        
        x = self.flatten(x)
        
        x = self.linear1(x) 
        x = self.activation(x)
        
        x = self.linear2(x)
        x = self.activation(x)
        
        x = self.linear3(x)
        x = self.activation(x)
        
        x = self.linear4(x)
        x = self.activation(x)
        
        x = self.linear5(x)
        x = torch.squeeze(x, dim=-1)
        return x
  
    def compute_loss_contrastive_divergence_positive(self, prediction_negative, prediction_positive):
        loss = torch.mean(prediction_negative) - torch.mean(prediction_positive)
        return loss

    def compute_loss_contrastive_divergence_negative(self, prediction_negative):
        loss = - torch.mean(prediction_negative)
        return loss

    def compute_loss_positive(self, prediction_negative, prediction_positive):
        loss = self.compute_loss_contrastive_divergence_positive(prediction_negative, prediction_positive)
        return loss

    def compute_loss_negative(self, prediction_negative):
        loss = self.compute_loss_contrastive_divergence_negative(prediction_negative)
        return loss

# ======================================================================
# encoder 
# ======================================================================
class encoder2(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        dim_latent: int=100,
        use_batch_norm: bool=True):
        
        super(encoder2, self).__init__()
       
        self.use_batch_norm = use_batch_norm
        self.activation     = nn.LeakyReLU()
        self.flatten        = nn.Flatten()
 
        self.conv1  = nn.Conv2d(in_channels=dim_channel*1, out_channels=dim_feature*1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2  = nn.Conv2d(in_channels=dim_feature*1, out_channels=dim_feature*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3  = nn.Conv2d(in_channels=dim_feature*2, out_channels=dim_feature*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4  = nn.Conv2d(in_channels=dim_feature*4, out_channels=dim_feature*8, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5  = nn.Conv2d(in_channels=dim_feature*8, out_channels=dim_feature*16, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.bn1    = nn.BatchNorm2d(dim_feature*1)
        self.bn2    = nn.BatchNorm2d(dim_feature*2)
        self.bn3    = nn.BatchNorm2d(dim_feature*4)
        self.bn4    = nn.BatchNorm2d(dim_feature*8)
        self.bn5    = nn.BatchNorm2d(dim_feature*16)

        self.linear1    = nn.Linear(in_features=dim_feature*16, out_features=dim_latent, bias=True)
        self.linear2    = nn.Linear(in_features=dim_feature*16, out_features=dim_latent, bias=True)

    def reparameterization(self, mu, log_var):
        sigma   = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z       = mu + sigma * epsilon
        return z

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.activation(x)

        x = self.conv5(x)
        if self.use_batch_norm:
            x = self.bn5(x)
        x = self.activation(x)
        
        x = self.flatten(x)
        
        mu      = self.linear1(x)
        log_var = self.linear2(x)
        z       = self.reparameterization(mu, log_var) 
        return mu, log_var, z

# ======================================================================
# decoder 
# ======================================================================
class decoder2(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        dim_latent: int=100,
        use_batch_norm: bool=True,
        activation_output: str='sigmoid'):

        super(decoder2, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.activation     = nn.LeakyReLU()
        self.unflatten      = nn.Unflatten(1, (dim_feature * 16, 1, 1))
        self.upsample       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if activation_output == 'sigmoid':
            self.activation_output = nn.Sigmoid()
        elif activation_output == 'tanh':
            self.activation_output = nn.Tanh()
        
        self.conv1  = nn.Conv2d(in_channels=dim_feature*16, out_channels=dim_feature*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2  = nn.Conv2d(in_channels=dim_feature*8, out_channels=dim_feature*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3  = nn.Conv2d(in_channels=dim_feature*4, out_channels=dim_feature*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4  = nn.Conv2d(in_channels=dim_feature*2, out_channels=dim_feature*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5  = nn.Conv2d(in_channels=dim_feature*1, out_channels=dim_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1    = nn.BatchNorm2d(dim_feature * 8)
        self.bn2    = nn.BatchNorm2d(dim_feature * 4)
        self.bn3    = nn.BatchNorm2d(dim_feature * 2)
        self.bn4    = nn.BatchNorm2d(dim_feature * 1)

        self.linear = nn.Linear(in_features=dim_latent, out_features=dim_feature*16, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.upsample(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.upsample(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.activation(x)
        x = self.upsample(x)
        
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.activation(x)
        x = self.upsample(x)
        
        x = self.conv5(x)
        x = self.activation_output(x)
        return x

# ======================================================================
# VAE 
# ======================================================================
class auto_encoder2(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        dim_latent: int=100,
        use_batch_norm: bool=True,
        activation_output: str='sigmoid'):

        super(auto_encoder2, self).__init__()
        
        self.encoder    = encoder2(dim_channel, dim_feature, dim_latent, use_batch_norm)
        self.decoder    = decoder2(dim_channel, dim_feature, dim_latent, use_batch_norm, activation_output)
   
    def forward(self, x):
        mu, log_var, z  = self.encoder(x)
        prediction      = self.decoder(z)
        return prediction, mu, log_var, z

    def compute_loss_data(self, prediction, target):
        criterion   = nn.MSELoss(reduction='mean')
        loss        = criterion(prediction, target)
        return loss

    def compute_loss_kld(self, mu, log_var):
        loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
        return loss

    def compute_loss(self, prediction, target, mu, log_var, weight_regular=1.0):
        loss_data   = self.compute_loss_data(prediction, target)
        loss_kld    = weight_regular * self.compute_loss_kld(mu, log_var)
        loss        = loss_data + loss_kld
        return loss, loss_data, loss_kld 


# ======================================================================
# encoder 
# ======================================================================
class encoder3(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        dim_latent: int=100,
        use_batch_norm: bool=True):
        
        super(encoder3, self).__init__()
       
        self.use_batch_norm = use_batch_norm
        self.activation     = nn.LeakyReLU()
        self.flatten        = nn.Flatten()
 
        self.conv1  = nn.Conv2d(in_channels=dim_channel*1, out_channels=dim_feature*1, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2  = nn.Conv2d(in_channels=dim_feature*1, out_channels=dim_feature*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3  = nn.Conv2d(in_channels=dim_feature*2, out_channels=dim_feature*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4  = nn.Conv2d(in_channels=dim_feature*4, out_channels=dim_feature*8, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5  = nn.Conv2d(in_channels=dim_feature*8, out_channels=dim_feature*16, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv6  = nn.Conv2d(in_channels=dim_feature*16, out_channels=dim_feature*32, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.bn1    = nn.BatchNorm2d(dim_feature*1)
        self.bn2    = nn.BatchNorm2d(dim_feature*2)
        self.bn3    = nn.BatchNorm2d(dim_feature*4)
        self.bn4    = nn.BatchNorm2d(dim_feature*8)
        self.bn5    = nn.BatchNorm2d(dim_feature*16)
        self.bn6    = nn.BatchNorm2d(dim_feature*32)


        self.linear1    = nn.Linear(in_features=dim_feature*32, out_features=dim_latent, bias=True)
        self.linear2    = nn.Linear(in_features=dim_feature*32, out_features=dim_latent, bias=True)

    def reparameterization(self, mu, log_var):
        sigma   = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z       = mu + sigma * epsilon
        return z

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.activation(x)

        x = self.conv5(x)
        if self.use_batch_norm:
            x = self.bn5(x)
        x = self.activation(x)
        
        x = self.conv6(x)
        if self.use_batch_norm:
            x = self.bn6(x)
        x = self.activation(x)

        x = self.flatten(x) # [100,1024]
        
        mu      = self.linear1(x) # latent [100,100]
        log_var = self.linear2(x) # latent [100,100]
        z       = self.reparameterization(mu, log_var)  # latent [100,100]
        return mu, log_var, z

# ======================================================================
# decoder 
# ======================================================================
class decoder3(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        dim_latent: int=100,
        use_batch_norm: bool=True,
        activation_output: str='tanh'):

        super(decoder3, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.activation     = nn.LeakyReLU()
        self.unflatten      = nn.Unflatten(1, (dim_feature * 32, 1, 1))
        self.upsample       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if activation_output == 'sigmoid':
            self.activation_output = nn.Sigmoid()
        elif activation_output == 'tanh':
            self.activation_output = nn.Tanh()
        
        self.conv1  = nn.ConvTranspose2d(in_channels=dim_feature*16*2, out_channels=dim_feature*8*2, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True)
        self.conv2  = nn.ConvTranspose2d(in_channels=dim_feature*8*2, out_channels=dim_feature*4*2, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True)
        self.conv3  = nn.ConvTranspose2d(in_channels=dim_feature*4*2, out_channels=dim_feature*2*2, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True)
        self.conv4  = nn.ConvTranspose2d(in_channels=dim_feature*2*2, out_channels=dim_feature*1*2, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True)
        self.conv5  = nn.ConvTranspose2d(in_channels=dim_feature*1*2, out_channels=dim_feature*1, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True)
        self.conv6  = nn.ConvTranspose2d(in_channels=dim_feature*1, out_channels=dim_channel, kernel_size=3, stride=2, output_padding=1, padding=1, bias=True)

        self.bn1    = nn.BatchNorm2d(dim_feature * 8 *2)
        self.bn2    = nn.BatchNorm2d(dim_feature * 4 *2)
        self.bn3    = nn.BatchNorm2d(dim_feature * 2 *2)
        self.bn4    = nn.BatchNorm2d(dim_feature * 1 *2)
        self.bn5    = nn.BatchNorm2d(dim_feature * 1 *2)

        self.linear = nn.Linear(in_features=dim_latent, out_features=dim_feature*32, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.activation(x)

        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.activation(x)
        
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.activation(x)
        
        x = self.conv5(x)
        if self.use_batch_norm:
            x = self.bn5(x)
        x = self.activation(x)

        x = self.conv6(x)
        x = self.activation_output(x)
        
        return x

# ======================================================================
# VAE 
# ======================================================================
class auto_encoder3(nn.Module):
    def __init__(self, 
        dim_channel: int=1, 
        dim_feature: int=32,
        dim_latent: int=100,
        use_batch_norm: bool=True,
        activation_output: str='sigmoid'):

        super(auto_encoder3, self).__init__()
        
        self.encoder    = encoder3(dim_channel, dim_feature, dim_latent, use_batch_norm)
        self.decoder    = decoder3(dim_channel, dim_feature, dim_latent, use_batch_norm, activation_output)
   
    def forward(self, x):
        mu, log_var, z  = self.encoder(x)
        prediction      = self.decoder(z)
        return prediction, mu, log_var, z

    def compute_loss_data(self, prediction, target):
        criterion   = nn.MSELoss(reduction='mean')
        loss        = criterion(prediction, target)
        return loss

    def compute_loss_kld(self, mu, log_var):
        loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
        return loss

    def compute_loss(self, prediction, target, mu, log_var, weight_regular=1.0):
        loss_data   = self.compute_loss_data(prediction, target)
        loss_kld    = weight_regular * self.compute_loss_kld(mu, log_var)
        loss        = loss_data + loss_kld
        return loss, loss_data, loss_kld 