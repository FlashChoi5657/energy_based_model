import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime, random
import csv, os, argparse, time
from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256
from ResNet import IGEBM, EnResnet, EnResnetSN
from models import CondEnergyModel, EnergyModel, UNet
from sn_gan import Discriminator
from utils import UNet_Energy_log, init_weights, Self_Energy_log, Custom_Dataset, SGLD
from utils import gaussian_noise, SS_denoise, to_numpy, SampleBuffer, get_dataloaders

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='base')
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--in_channel", type=int, default=3)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--dim_feature", type=int, default=128)

parser.add_argument("--lr_unet", type=float, default=0.0001)
parser.add_argument("--lr_energy_model", type=float, default=0.00005)
parser.add_argument("--lr_langevin_max", type=float, default=0.02)
parser.add_argument("--lr_langevin_min", type=float, default=0.01)
parser.add_argument("--iter_LD", type=int, default=200)
parser.add_argument("--noise_max", type=float, default=0.5)
parser.add_argument("--sigma_noise", type=float, default=0.25)

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_plot", type=int, default=20)
parser.add_argument("--terminate", type=int, default=100)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--data_size", type=int, default=1500)
parser.add_argument("--subset", action="store_true", default=True)

parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--use_unpaired", action="store_true", default=False)
parser.add_argument("--init_noise_decay", type=float, default=1.0)
# parser.add_argument("--use_gp", action="store_true", default=True)
# parser.add_argument("--use_energy_reg", action="store_true", default=False)
parser.add_argument("--energy_reg_weight", type=float, default=0.0005)
parser.add_argument("--use_en_L2_reg", action="store_true", default=False)
parser.add_argument("--L2", type=float, default=0.1)

parser.add_argument("--gpu_devices", type=int, default=0)
parser.add_argument("--model_load", type=bool, default=False)

args = parser.parse_args()

# ======================================================================
# Options
# ======================================================================
dir_work              = '/nas/users/minhyeok/energy_based_model/experiments'
name                  = args.name
cuda_device           = args.gpu_devices
NGPU                  = torch.cuda.device_count()
device                = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
number_epoch          = args.epochs
lr_energy_model       = args.lr_energy_model
lr_unet               = args.lr_unet
use_unpaired          = args.use_unpaired
in_channel            = args.in_channel
dim_feature           = args.dim_feature
dim_output            = 1
# use_gp                = args.use_gp
seed                  = args.seed
list_lr_LD            = np.linspace(args.lr_langevin_max, args.lr_langevin_min, num=number_epoch, endpoint=True)
pl.seed_everything(seed)

dir_data = '/hdd1/dataset/'
# dir_data    = '/nas/dataset/users/minhyeok/recon'
dir_dataset = args.dataset

# ======================================================================
# Path for the results
# ======================================================================
dir_figure = os.path.join(dir_work, 'figure')
dir_option = os.path.join(dir_work, 'option')
dir_result = os.path.join(dir_work, 'result')
dir_model  = os.path.join(dir_work, 'model')

date_stamp = datetime.datetime.now().strftime('%Y_%m_%d')

path_figure = os.path.join(dir_figure, dir_dataset)
path_option = os.path.join(dir_option, dir_dataset)
path_result = os.path.join(dir_result, dir_dataset)
path_model  = os.path.join(dir_model, 'MDSM')

date_figure = os.path.join(path_figure, date_stamp)
date_option = os.path.join(path_option, date_stamp)
date_result = os.path.join(path_result, date_stamp)

if not os.path.exists(date_figure):  os.makedirs(date_figure, exist_ok=True)
if not os.path.exists(date_option):  os.makedirs(date_option, exist_ok=True)
if not os.path.exists(date_result):  os.makedirs(date_result, exist_ok=True)
if not os.path.exists(path_model):  os.makedirs(path_model, exist_ok=True)

# ======================================================================
# Dataloaders 
# ======================================================================

transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.img_size),
    # torchvision.transforms.RandomCrop(32, padding=4),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # torchvision.transforms.Normalize((0.5, ), (0.5, ))
    ])

dataloader_train, dataloader_test= get_dataloaders(dir_data, args.dataset, args.img_size, 
                                        args.batch_size, train_size = args.data_size,
                                        transform=transforms_train,
                                        parallel=False)

PSNR = PeakSignalNoiseRatio().to(device)
energy = EnResnet(in_channel, dim_feature, activation='lrelu')
num_params = sum(p.numel() for p in energy.parameters() if p.requires_grad)
print('The number of parameters of energy_model is', format(num_params, ','))

energy = energy.to(device)

optim_en = torch.optim.Adam(energy.parameters(), lr=lr_energy_model, betas=(args.b1, 0.99))
# sched_optim_energy = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda = lambda epoch: 0.95 ** epoch)
scheduler_en = torch.optim.lr_scheduler.StepLR(optim_en, step_size=10, gamma=0.97)
sigmas_np = np.linspace(0.05, args.noise_max, args.batch_size)
sigmas = torch.Tensor(sigmas_np).view((args.batch_size, 1, 1, 1)).to(device)

loss_en                    = np.zeros(number_epoch)
# val_psnr_ori_mean          = np.zeros(number_epoch)
val_psnr_mean              = np.zeros(number_epoch)
val_psnr_langevin_mean     = np.zeros(number_epoch)
val_loss_unet_mean         = np.zeros(number_epoch)

for i in range(number_epoch):
    val_loss = list()
    val_psnr = list()
    
    start = time.time()
    
    for j, (image, _) in enumerate(iter(dataloader_train)):
    
        image = image.to(device)
        pred = image + sigmas*torch.randn_like(image)

        # energy.eval()
        # pred = SGLD(noisy, energy, args.iter_LD, list_lr_LD[i])

        energy.train()
        pred = pred.requires_grad_()
        E = energy(pred).sum()
        grad_x = torch.autograd.grad(E,pred,create_graph=True)[0]
        pred.detach()

        optim_en.zero_grad()

        LS_loss = ((((image-pred)/sigmas/args.L2 + grad_x/sigmas)**2)/args.batch_size).sum()

        LS_loss.backward()
        optim_en.step()
        # lr = scheduler_en.get_last_lr()[0]
        # scheduler_en.step()
        val_loss.append(LS_loss.item())
    
    energy.eval()
    update = SS_denoise(pred, energy, np.sqrt(args.L2))
    val_init = to_numpy(PSNR(pred, image))
    val_psnr = to_numpy(PSNR(update, image))
    val_psnr_mean[i] = val_psnr
    loss_en[i] = np.mean(val_loss)
    
    time_spent = time.time() - start
    print('Epochs {}/{} | loss: {:.5f} | noisy: {:.5f} | output: {:.5f} | time {:.1f}'.format(i+1, number_epoch, loss_en[i], val_init, val_psnr, time_spent))
    
    fSize, col, row = 3, 8, 3
    if i % args.save_plot == (args.save_plot-1):
        fig, ax = plt.subplots(row, col, figsize = (fSize * 15, fSize * col))
        for k in range(col):
            rand = random.randrange(args.batch_size)
            ax[0][k].set_title('Train (Clean)')
            ax[0][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
            ax[1][k].set_title('Train (Noisy)')
            ax[1][k].imshow(torchvision.utils.make_grid(pred[rand].detach().cpu(), normalize=True).permute(1,2,0))
            ax[2][k].set_title('Train (Langevin)')
            ax[2][k].imshow(torchvision.utils.make_grid(update[rand].detach().cpu(), normalize=True).permute(1,2,0))
        time_ = datetime.datetime.now().strftime('%HH_%MM')

        plt.tight_layout()
        fig.savefig(f'{date_figure}/{name}_epoch:{i}_{time_}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        

torch.save(energy,f'{path_model}/{date_stamp}.pt')


energy.eval()

# ======================================================================
# Testing results
# ======================================================================
(image_test, _) = next(iter(dataloader_test))
image_test = image_test.to(device)
image_noise_test = image_test + sigmas * torch.randn_like(image_test)
# image_noise_test = gaussian_noise(image_test, args.sigma_noise)
# image_noise_test = image_noise_test.to(device)
# y_test           = generate_Y0(image_noise_test, Y0_type)
y_update_test = SGLD(image_noise_test, energy, args.iter_LD, list_lr_LD[-1])

# ---------------------------------------------------clip_grad-------------
test_self_data_psnr     = np.zeros(len(dataloader_test))
test_energy_data_psnr = np.zeros(len(dataloader_test))

for j, (image_test_, _) in enumerate(iter(dataloader_test)):
  image_test_       = image_test_.to(device)
  # image_noise_test_ = gaussian_noise(image_test_, args.sigma_noise)
  image_noise_test_ = image_test_ + sigmas * torch.randn_like(image_test_).to(device)
  test_self_data_psnr[j] = to_numpy(PSNR(image_noise_test_, image_test_)).mean()
  # y_test_           = generate_Y0(image_noise_test_, Y0_type)
  y_update_test_    = SGLD(image_noise_test_, energy, args.iter_LD, list_lr_LD[-1])
  test_energy_data_psnr[j] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

nRow  = 7
nCol  = 6

fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('UNet Model')
ax[0][0].plot(val_loss_unet_mean[:i], color='red', label='Loss')
ax[0][0].legend()

ax[0][1].set_title('Energy Model')
ax[0][1].plot(loss_en[:i], color='red', label='Loss')
ax[0][1].legend()

ax[0][2].set_title('Train PSNR')
ax[0][2].plot(val_psnr_mean[:i], color='red', label='output')
ax[0][2].legend()

bplot_colors = ['pink', 'lightgreen']

ax[0][3].set_title('Energy Data PSNR')
ax[0][3].yaxis.grid(True)
bplot1 = ax[0][3].boxplot([test_self_data_psnr,
                          test_energy_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot1['boxes'], bplot_colors):
  patch.set_facecolor(color)

  
for k in range(nCol):
  rand = random.randrange(args.batch_size)
  ax[1][k].set_title('Train (Input)')
  ax[1][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[2][k].set_title('Train (Noisy)')
  ax[2][k].imshow(torchvision.utils.make_grid(pred[rand].detach().cpu(), normalize=True).permute(1,2,0))

  ax[3][k].set_title('Train (Langevin)')
  ax[3][k].imshow(torchvision.utils.make_grid(update[rand].detach().cpu(), normalize=True).permute(1,2,0))

  ax[4][k].set_title('Test (Input)')
  ax[4][k].imshow(torchvision.utils.make_grid(image_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[5][k].set_title('Test (Noisy)')
  ax[5][k].imshow(torchvision.utils.make_grid(image_noise_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[6][k].set_title('Test (Langevin)')
  ax[6][k].imshow(torchvision.utils.make_grid(y_update_test[rand].detach().cpu(), normalize=True).permute(1,2,0))

time_1 = datetime.datetime.now().strftime('%HH_%MM')
plt.tight_layout()
fig.savefig(f'{date_figure}/last_{name}_{time_1}.png', bbox_inches='tight', dpi=600)
plt.close(fig)