import torch
import torchvision
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import os
import copy
import argparse
import configparser
import sys
import datetime 
from tqdm import tqdm
import csv
from torchmetrics import PeakSignalNoiseRatio
import pytorch_lightning as pl
from torchvision.utils import save_image
import random
import network_models as models
import function_losses as losses


# ======================================================================
# take options 
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dir_work", type=str, default='/home/hong/work/energy_based_model')
parser.add_argument("--device_cuda", type=int, default=0)

parser.add_argument("--model_conv", type=str, default='conv_double_resnet')
parser.add_argument("--model_activation", type=str, default='leakyrelu')
parser.add_argument("--model_output", type=str, default='sigmoid')
parser.add_argument("--model_use_batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument("--model_use_skip", type=eval, default=False, choices=[True, False])
parser.add_argument("--model_use_dual_input", type=eval, default=False, choices=[True, False])
parser.add_argument("--model_dim_feature", type=int, default=16)
parser.add_argument("--model_dim_latent", type=int, default=100)

parser.add_argument("--data_name", type=str, default='MNIST')
parser.add_argument("--data_use_all", type=eval, default=False, choices=[True, False])
parser.add_argument("--data_label_subset", type=int, default=5)
parser.add_argument("--data_channel", type=int, default=1)
parser.add_argument("--data_height", type=int, default=32)
parser.add_argument("--data_width", type=int, default=32)
parser.add_argument("--data_noise_sigma", type=float, default=0.5)

parser.add_argument("--optim_option", type=str, default='adam')
parser.add_argument("--optim_length_epoch", type=int, default=100)
parser.add_argument("--optim_size_batch", type=int, default=100)
parser.add_argument("--optim_lr_model", type=float, default=0.01)
parser.add_argument("--optim_lr_energy", type=float, default=0.01)
parser.add_argument("--optim_lr_data", type=float, default=0.01)
parser.add_argument("--optim_lr_langevin", type=float, default=0.0001)
parser.add_argument("--optim_length_langevin", type=int, default=10)
parser.add_argument("--optim_weight_gradient", type=float, default=0.0001)
parser.add_argument("--optim_weight_regular", type=float, default=0.0001)

args = parser.parse_args()

# ======================================================================
# assign options
# ======================================================================
dir_work                = args.dir_work
device_cuda             = args.device_cuda

model_conv              = args.model_conv
model_activation        = args.model_activation
model_output            = args.model_output
model_use_batch_norm    = args.model_use_batch_norm
model_use_skip          = args.model_use_skip
model_use_dual_input    = args.model_use_dual_input
model_dim_feature       = args.model_dim_feature
model_dim_latent        = args.model_dim_latent

data_name               = args.data_name.upper()
data_use_all            = args.data_use_all
data_label_subset       = args.data_label_subset 
data_channel            = args.data_channel
data_height             = args.data_height
data_width              = args.data_width
data_noise_sigma        = args.data_noise_sigma

optim_option            = args.optim_option
optim_length_epoch      = args.optim_length_epoch
optim_size_batch        = args.optim_size_batch
optim_lr_model          = args.optim_lr_model
optim_lr_energy         = args.optim_lr_energy
optim_lr_data           = args.optim_lr_data
optim_lr_langevin       = args.optim_lr_langevin
optim_length_langevin   = args.optim_length_langevin
optim_weight_gradient   = args.optim_weight_gradient
optim_weight_regular    = args.optim_weight_regular

# ======================================================================
# path for the results
# ======================================================================
now         = datetime.datetime.now()
date_stamp  = now.strftime('%Y_%m_%d') 
time_stamp  = now.strftime('%H_%M_%S') 

dir_figure  = os.path.join(dir_work, 'figure')
dir_option  = os.path.join(dir_work, 'option')
dir_result  = os.path.join(dir_work, 'result')
dir_model   = os.path.join(dir_work, 'model')

path_figure = os.path.join(dir_figure, data_name)
path_option = os.path.join(dir_option, data_name)
path_result = os.path.join(dir_result, data_name)
path_model  = os.path.join(dir_model, data_name)

date_figure = os.path.join(path_figure, date_stamp)
date_option = os.path.join(path_option, date_stamp)
date_result = os.path.join(path_result, date_stamp)
date_model  = os.path.join(path_model, date_stamp)

file_figure = os.path.join(date_figure, '{}.png'.format(time_stamp))
file_option = os.path.join(date_option, '{}.ini'.format(time_stamp))
file_result = os.path.join(date_result, '{}.csv'.format(time_stamp))
file_model  = os.path.join(date_model, '{}.pth'.format(time_stamp))

if not os.path.exists(dir_figure):
    os.mkdir(dir_figure)
if not os.path.exists(dir_option):
    os.mkdir(dir_option)
if not os.path.exists(dir_result):
    os.mkdir(dir_result)
if not os.path.exists(dir_model):
    os.mkdir(dir_model)
if not os.path.exists(path_figure):
    os.mkdir(path_figure)
if not os.path.exists(path_option):
    os.mkdir(path_option)
if not os.path.exists(path_result):
    os.mkdir(path_result)
if not os.path.exists(path_model):
    os.mkdir(path_model)
if not os.path.exists(date_figure):
    os.mkdir(date_figure)
if not os.path.exists(date_option):
    os.mkdir(date_option)
if not os.path.exists(date_result):
    os.mkdir(date_result)
if not os.path.exists(date_model):
    os.mkdir(date_model)

# ======================================================================
# device
# ======================================================================
device = torch.device(f'cuda:{device_cuda}' if torch.cuda.is_available() else 'mps')

# ======================================================================
# random seed
# ======================================================================
pl.seed_everything(0)

# ======================================================================
# dataset 
# ======================================================================
dir_data = os.path.join(dir_work, 'data')

transform = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize([data_height, data_width]),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Lambda(lambda t: (t - torch.mean(t)) / torch.std(t)) # mean 0, std 1
    # torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
])

# the name of the dataset is used as upper case
if data_name == 'MNIST':
    dataset         = torchvision.datasets.MNIST(dir_data, transform=transform, train=True, download=True)
    dataset_test    = torchvision.datasets.MNIST(dir_data, transform=transform, train=False, download=True)

elif data_name == 'CIFAR10':
    dataset                 = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=True, download=True)
    dataset.data            = np.array(dataset.data)
    dataset.targets         = np.array(dataset.targets)
    dataset_test            = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=False, download=True)
    dataset_test.data       = np.array(dataset_test.data)
    dataset_test.targets    = np.array(dataset_test.targets)

elif data_name == 'CELEBA':
    dataset = torchvision.datasets.CelebA(dir_data, transform=transform, download=True)
   
if data_name == 'MNIST' or data_name == 'CIFAR10': 
    if not data_use_all:
        idx_label               = (dataset.targets==data_label_subset)
        dataset.data            = dataset.data[idx_label]
        dataset.targets         = dataset.targets[idx_label]
        
        idx_label               = (dataset_test.targets==data_label_subset)
        dataset_test.data       = dataset_test.data[idx_label]
        dataset_test.targets    = dataset_test.targets[idx_label]

    num_data_real       = len(dataset)
    number_data_real    = 5000
    dataset.data        = dataset.data[0:number_data_real]
    dataset.targets     = dataset.targets[0:number_data_real]

dataloader      = torch.utils.data.DataLoader(dataset=dataset, batch_size=optim_size_batch*2, drop_last=True, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=optim_size_batch, drop_last=True, shuffle=True)


model_conv              = args.model_conv
model_activation        = args.model_activation
model_output            = args.model_output
model_use_batch_norm    = args.model_use_batch_norm
model_use_skip          = args.model_use_skip
model_use_dual_input    = args.model_use_dual_input
model_dim_feature       = args.model_dim_feature
model_dim_latent        = args.model_dim_latent

data_name               = args.data_name.upper()
data_use_all            = args.data_use_all
data_label_subset       = args.data_label_subset 
data_channel            = args.data_channel
data_height             = args.data_height
data_width              = args.data_width

optim_option            = args.optim_option
optim_length_epoch      = args.optim_length_epoch
optim_size_batch        = args.optim_size_batch
optim_lr_model          = args.optim_lr_model
optim_lr_energy         = args.optim_lr_energy
optim_lr_langevin       = args.optim_lr_langevin
optim_length_langevin   = args.optim_length_langevin
optim_weight_gradient   = args.optim_weight_gradient

# ======================================================================
# model 
# ======================================================================
model = models.auto_encoder2(
            dim_channel=data_channel,
            dim_feature=model_dim_feature,
            dim_latent=model_dim_latent,
            use_batch_norm=model_use_batch_norm, 
            activation_output=model_output).to(device)
'''
energy = models.energy2(
            dim_channel=data_channel,
            dim_feature=model_dim_feature,
            use_batch_norm=model_use_batch_norm,
            use_dual_input=model_use_dual_input).to(device)
'''

energy = models.energy2(
            dim_channel=data_channel,
            dim_feature=model_dim_feature,
            use_batch_norm=False,
            use_dual_input=model_use_dual_input).to(device)
             
# ======================================================================
# optimizer 
# ======================================================================
if optim_option.lower() == 'sgd':
    optim_model     = torch.optim.SGD(model.parameters(), lr=optim_lr_model)
    optim_energy    = torch.optim.SGD(energy.parameters(), lr=optim_lr_energy)
elif optim_option.lower() == 'adam':
    optim_model     = torch.optim.Adam(model.parameters(), lr=optim_lr_model)
    optim_energy    = torch.optim.Adam(energy.parameters(), lr=optim_lr_energy)
elif optim_option.lower() == 'adamw':
    optim_model     = torch.optim.AdamW(model.parameters(), lr=optim_lr_model)
    optim_energy    = torch.optim.AdamW(energy.parameters(), lr=optim_lr_energy)

#scheduler = torch.optim.lr_scheduler.OneCycleLR(optim_energy, max_lr=lr_energy, div_factor=10, final_div_factor=100, total_steps=length_epoch)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_energy, length_epoch) 

#scheduler_model     = torch.optim.lr_scheduler.OneCycleLR(optim_model, max_lr=optim_lr_model, steps_per_epoch=optim_length_epoch, epochs=optim_length_epoch)
#scheduler_energy    = torch.optim.lr_scheduler.OneCycleLR(optim_energy, max_lr=optim_lr_energy, steps_per_epoch=optim_length_epoch, epochs=optim_length_epoch)

#scheduler_model     = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, optim_length_epoch)
#scheduler_energy    = torch.optim.lr_scheduler.CosineAnnealingLR(optim_energy, optim_length_epoch)

#scheduler_model     = torch.optim.lr_scheduler.StepLR(optim_model, step_size=10, gamma=0.9)
#scheduler_energy    = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=10, gamma=0.9)

scheduler_model     = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_model, factor=0.0001, patience=10, mode='min')
scheduler_energy    = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_energy, factor=0.0001, patience=10, mode='min')

# ======================================================================
# evaluation 
# ======================================================================
psnr = PeakSignalNoiseRatio().to(device)

# ======================================================================
# training 
# ======================================================================
val_loss_model_mean     = np.zeros(optim_length_epoch)
val_loss_model_std      = np.zeros(optim_length_epoch)
val_loss_energy_mean    = np.zeros(optim_length_epoch)
val_loss_energy_std     = np.zeros(optim_length_epoch)
val_psnr_mean           = np.zeros(optim_length_epoch)
val_psnr_std            = np.zeros(optim_length_epoch)
val_psnr_update_mean    = np.zeros(optim_length_epoch)
val_psnr_update_std     = np.zeros(optim_length_epoch)
 
model.train()
energy.train()

for i in range(optim_length_epoch):
    val_loss_model  = []
    val_loss_energy = []
    val_psnr        = []
    val_psnr_update = []

    for j, (image, _) in enumerate(dataloader):
        data, real  = torch.split(image, optim_size_batch, dim=0)
        
        noise       = torch.randn_like(data)
        data_noise  = data + data_noise_sigma * noise
        
        noise       = torch.randn_like(data)
        real_noise  = real + data_noise_sigma * noise
     
        data        = data.to(device) 
        real        = real.to(device) 
        data_noise  = data_noise.to(device)
        real_noise  = real_noise.to(device)
        
        # -------------------------------------------------------------------
        # predictions
        # -------------------------------------------------------------------         
        pred, mu, log_var, z = model(data_noise)
         
        # -------------------------------------------------------------------
        # interpolation
        # -------------------------------------------------------------------         
        alpha   = torch.rand(optim_size_batch, 1, 1, 1)
        alpha   = alpha.expand_as(real).to(device)
        interp  = alpha * real.data + (1 - alpha) * pred.data
        interp  = Parameter(interp, requires_grad=True)

        # -------------------------------------------------------------------
        # predictions
        # -------------------------------------------------------------------
        if model_use_dual_input:
            energy_positive = energy(real_noise, real)        
            energy_negative = energy(data_noise, pred)        
            energy_interp   = energy(interp, interp)
        else:
            energy_positive = energy(real)        
            energy_negative = energy(pred)        
            energy_interp   = energy(interp)
        
        # -------------------------------------------------------------------
        # update energy model 
        # -------------------------------------------------------------------         
        optim_energy.zero_grad()
        loss_positive   = energy.compute_loss_positive(energy_negative, energy_positive)
        loss_gradient   = losses.compute_gradient_penalty(interp, energy_interp)
        loss_energy     = loss_positive + optim_weight_gradient * loss_gradient
        loss_energy.backward()
        optim_energy.step()
        scheduler_energy.step(loss_energy)
         
        # -------------------------------------------------------------------
        # update input fake 
        # -------------------------------------------------------------------
        pred_update = Parameter(pred, requires_grad=True) 
        
        for k in range(optim_length_langevin): 

            if model_use_dual_input:
                energy_negative = energy(data_noise, pred_update)
            else:
                energy_negative = energy(pred_update)
                
            loss_negative = energy.compute_loss_negative(energy_negative)
            loss_negative.backward()
            noise = torch.randn_like(pred_update.data)    # N(mean=0, std=1)
            pred_update.data = pred_update.data - optim_lr_data * pred_update.grad + optim_lr_langevin * noise
            pred_update.grad.detach_()
            pred_update.grad.zero_() 

        # -------------------------------------------------------------------
        # update model 
        # -------------------------------------------------------------------         
        optim_model.zero_grad()
        pred, mu, log_var, z = model(data_noise) 
        loss_model, loss_data, loss_regular = model.compute_loss(pred, pred_update.detach(), mu, log_var, optim_weight_regular) 
        loss_model.backward()
        optim_model.step()        
        scheduler_model.step(loss_model)

        value_psnr          = psnr(pred.data, data).detach().cpu().numpy().mean()
        value_psnr_update   = psnr(pred_update.data, data).detach().cpu().numpy().mean()
            
        val_loss_model.append(loss_model.item()) 
        val_loss_energy.append(loss_energy.item()) 
        val_psnr.append(value_psnr)
        val_psnr_update.append(value_psnr_update)
    
    if i % 10 == 0:
        dir_log             = os.path.join(dir_work, 'log')
        file_pred           = os.path.join(dir_log, 'image/pred.png')
        file_pred_update    = os.path.join(dir_log, 'image/pred_update.png')
        # file_pred           = os.path.join(dir_log, 'image/{:03d}.png'.format(i))
        # file_pred_update    = os.path.join(dir_log, 'image/{:03d}_update.png'.format(i))
    
        save_image(pred.data[:25], file_pred, nrow=5, normalize=True)
        save_image(pred_update.data[:25], file_pred_update, nrow=5, normalize=True)
        
    
    val_loss_model_mean[i]  = np.mean(val_loss_model)
    val_loss_model_std[i]   = np.std(val_loss_model)
    val_loss_energy_mean[i] = np.mean(val_loss_energy)
    val_loss_energy_std[i]  = np.std(val_loss_energy)
    val_psnr_mean[i]        = np.mean(val_psnr)
    val_psnr_std[i]         = np.std(val_psnr)
    val_psnr_update_mean[i] = np.mean(val_psnr_update)
    val_psnr_update_std[i]  = np.std(val_psnr_update)

    log = '[%4d/%4d] loss(model)=%8.7f, loss(energy)=%8.7f, psnr=%4.2f, psnr(update)=%4.2f' % (i, optim_length_epoch, val_loss_model_mean[i], val_loss_energy_mean[i], val_psnr_mean[i], val_psnr_update_mean[i])
    print(log, flush=True)
    
    if np.isnan(val_loss_model_mean[i]) or np.isnan(val_loss_energy_mean[i]) or val_psnr_mean[i] < 3:
        sys.exit('error')

# -------------------------------------------------------------------
# save the models
# -------------------------------------------------------------------          
torch.save({
    'state_dict_model'      : model.state_dict(),
    'state_dict_energy'     : energy.state_dict(),
    'model_conv'            : model_conv,
    'model_activation'      : model_activation,
    'model_output'          : model_output,
    'model_use_batch_norm'  : model_use_batch_norm,
    'model_use_skip'        : model_use_skip,
    'model_use_dual_input'  : model_use_dual_input,
    'model_dim_feature'     : model_dim_feature,
    'model_dim_latent'      : model_dim_latent,
    'data_name'             : data_name,
    'data_use_all'          : data_use_all,
    'data_label_subset'     : data_label_subset,
    'data_channel'          : data_channel,
    'data_height'           : data_height,
    'data_width'            : data_width,
    'data_noise_sigma'      : data_noise_sigma,
    'optim_option'          : optim_option,
    'optim_length_epoch'    : optim_length_epoch,
    'optim_size_batch'      : optim_size_batch,
    'optim_lr_model'        : optim_lr_model,
    'optim_lr_energy'       : optim_lr_energy,
    'optim_lr_data'         : optim_lr_data,
    'optim_lr_langevin'     : optim_lr_langevin,
    'optim_length_langevin' : optim_length_langevin,
    'optim_weight_gradient' : optim_weight_gradient,
    'optim_weight_regular'  : optim_weight_regular,
}, file_model)


# -------------------------------------------------------------------
# save the options
# -------------------------------------------------------------------         
with open(file_option, 'w') as f:
    f.write('{}: {}\n'.format('model_conv', model_conv))
    f.write('{}: {}\n'.format('model_activation', model_activation))
    f.write('{}: {}\n'.format('model_output', model_output))
    f.write('{}: {}\n'.format('model_use_batch_norm', model_use_batch_norm))
    f.write('{}: {}\n'.format('model_use_skip', model_use_skip))
    f.write('{}: {}\n'.format('model_use_dual_input', model_use_dual_input))
    f.write('{}: {}\n'.format('model_dim_feature', model_dim_feature))
    f.write('{}: {}\n'.format('model_dim_latent', model_dim_latent))
    f.write('{}: {}\n'.format('data_name', data_name))
    f.write('{}: {}\n'.format('data_use_all', data_use_all))
    f.write('{}: {}\n'.format('data_label_subset', data_label_subset))
    f.write('{}: {}\n'.format('data_channel', data_channel))
    f.write('{}: {}\n'.format('data_height', data_height))
    f.write('{}: {}\n'.format('data_width', data_width))
    f.write('{}: {}\n'.format('data_noise_sigma', data_noise_sigma))
    f.write('{}: {}\n'.format('optim_option', optim_option))
    f.write('{}: {}\n'.format('optim_length_epoch', optim_length_epoch))
    f.write('{}: {}\n'.format('optim_size_batch', optim_size_batch))
    f.write('{}: {}\n'.format('optim_lr_model', optim_lr_model))
    f.write('{}: {}\n'.format('optim_lr_energy', optim_lr_energy))
    f.write('{}: {}\n'.format('optim_lr_data', optim_lr_data))
    f.write('{}: {}\n'.format('optim_lr_langevin', optim_lr_langevin))
    f.write('{}: {}\n'.format('optim_length_langevin', optim_length_langevin))
    f.write('{}: {}\n'.format('optim_weight_gradient', optim_weight_gradient))
    f.write('{}: {}\n'.format('optim_weight_regular', optim_weight_regular))
f.close()
    
# -------------------------------------------------------------------
# save training results
# -------------------------------------------------------------------         
data        = data.detach().cpu().numpy().squeeze()
data_noise  = data_noise.detach().cpu().numpy().squeeze()
pred        = pred.detach().cpu().numpy().squeeze()
pred_update = pred_update.detach().cpu().numpy().squeeze()

# -------------------------------------------------------------------
# save training results
# -------------------------------------------------------------------         
nRow    = 5 
nCol    = 5
fSize   = 3

fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('loss (model)')
ax[0][0].plot(val_loss_model_mean, color='red')
ax[0][0].fill_between(list(range(optim_length_epoch)), val_loss_model_mean-val_loss_model_std, val_loss_model_mean+val_loss_model_std, color='blue', alpha=0.2)

ax[0][1].set_title('loss (energy)')
ax[0][1].plot(val_loss_energy_mean, color='red')
ax[0][1].fill_between(list(range(optim_length_epoch)), val_loss_energy_mean-val_loss_energy_std, val_loss_energy_mean+val_loss_energy_std, color='blue', alpha=0.2)

ax[0][2].set_title('psnr')
ax[0][2].plot(val_psnr_mean, color='blue', label='y_0')
ax[0][2].plot(val_psnr_update_mean, color='red', label='y_n')
ax[0][2].legend()

for i in range(nCol):
    ax[1][i].set_title('clean')
    ax[1][i].imshow(data[i])

for i in range(nCol):
    ax[2][i].set_title('noisy')
    ax[2][i].imshow(data_noise[i])

for i in range(nCol):
    ax[3][i].set_title('y_0')
    ax[3][i].imshow(pred[i])

for i in range(nCol):
    ax[4][i].set_title('y_n')
    ax[4][i].imshow(pred_update[i])

plt.tight_layout()
fig.savefig(file_figure, bbox_inches='tight', dpi=300)
plt.close(fig)
