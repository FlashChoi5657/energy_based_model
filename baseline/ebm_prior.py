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
from ResNet import IGEBM, EnResnet
from models import CondEnergyModel, EnergyModel, UNet
from sn_gan import Discriminator
from utils import UNet_Energy_log, Self_Energy_log, Custom_Dataset, SGLD, SGLD_
from utils import gaussian_noise, calculate_norm, to_numpy, SampleBuffer, get_dataloaders

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='prior')
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--in_channel", type=int, default=3)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--dim_feature", type=int, default=128)

parser.add_argument("--lr_unet", type=float, default=0.0001)
parser.add_argument("--lr_energy_model", type=float, default=0.000001)
parser.add_argument("--lr_langevin_min", type=float, default=0.000000002)
parser.add_argument("--lr_langevin_max", type=float, default=0.00000008)
parser.add_argument("--iter_langevin", type=int, default=20)
parser.add_argument("--noise_max", type=float, default=0.2)
parser.add_argument("--sigma_noise", type=float, default=0.1)

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_plot", type=int, default=20)
parser.add_argument("--terminate", type=int, default=100)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data_size", type=int, default=1000)
parser.add_argument("--subset", action="store_true", default=True)

parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
parser.add_argument("--use_unpaired", action="store_true", default=False)
parser.add_argument("--regular_data", type=float, default=0.0)
parser.add_argument("--init_noise_decay", type=float, default=1.0)
# parser.add_argument("--use_gp", action="store_true", default=True)
# parser.add_argument("--use_energy_reg", action="store_true", default=False)
parser.add_argument("--energy_reg_weight", type=float, default=0.0005)
parser.add_argument("--use_en_L2_reg", action="store_true", default=False)
parser.add_argument("--energy_L2_reg_weight", type=float, default=0.0)

parser.add_argument("--gpu_devices", type=int, default=0)
parser.add_argument("--model_load", type=bool, default=False)

args = parser.parse_args()

# ======================================================================
# Options
# ======================================================================
# curr_dir_path         = os.path.join(os.path.dirname(os.path.abspath('')), '..', 'experiments')
dir_work              = '/nas/users/minhyeok/energy_based_model/experiments'
name                  = args.name
cuda_device           = args.gpu_devices
NGPU                  = torch.cuda.device_count()
device                = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
number_epoch          = args.epochs
sched_step_size       = number_epoch // 6
sched_gamma           = 0.97
lr_energy_model       = args.lr_energy_model
lr_unet               = args.lr_unet
number_step_langevin  = args.iter_langevin
regular_data          = args.regular_data
L2_w                  = args.energy_L2_reg_weight
use_unpaired          = args.use_unpaired
in_channel            = args.in_channel
dim_feature           = args.dim_feature
dim_output            = 1
# use_gp                = args.use_gp
seed                  = args.seed
list_lr_langevin      = np.linspace(args.lr_langevin_max, args.lr_langevin_min, num=number_epoch, endpoint=True)
pl.seed_everything(seed)

# ======================================================================
# Dataset 
# ======================================================================
# dir_data    = os.path.join(dir_work, 'data')
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
path_model  = os.path.join(dir_model, dir_dataset)

date_figure = os.path.join(path_figure, date_stamp)
date_option = os.path.join(path_option, date_stamp)
date_result = os.path.join(path_result, date_stamp)
date_model  = os.path.join(path_model, date_stamp)

if not os.path.exists(date_figure):  os.makedirs(date_figure, exist_ok=True)
if not os.path.exists(date_option):  os.makedirs(date_option, exist_ok=True)
if not os.path.exists(date_result):  os.makedirs(date_result, exist_ok=True)
if not os.path.exists(date_model):  os.makedirs(date_model, exist_ok=True)

# ======================================================================
# Dataloaders 
# ======================================================================

transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.img_size),
    # torchvision.transforms.RandomCrop(32, padding=4),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
    ])
# train_set = Custom_Dataset(dir_data, transform=transforms_train)
# test_set = Custom_Dataset(dir_data, transform=transforms_train, train=False)
# dataloader_train = DataLoader(dataset=train_set, batch_size=args.batch_size, drop_last=True, shuffle=True)
# dataloader_test = DataLoader(dataset=test_set, batch_size=args.batch_size, drop_last=True, shuffle=False)
dataloader_train, dataloader_test = get_dataloaders(dir_data, args.dataset, args.img_size, 
                                            args.batch_size, train_size = args.data_size,
                                            transform=transforms_train,
                                            use_subset = args.subset, use_label=1,
                                            parallel=False)
# ======================================================================
# Evaluation
# ======================================================================
PSNR = PeakSignalNoiseRatio().to(device)

# ======================================================================
# Instantiations
# ======================================================================

# energy = Discriminator(in_channel, dim_feature, activation='swish')
energy = EnResnet(in_channel, dim_feature, activation='lrelu')
# energy = IGEBM(args.in_channel, args.dim_feature, 1)

# unet = UNet(in_channel, dim_feature, in_channel)
num_params = sum(p.numel() for p in energy.parameters() if p.requires_grad)
print('The number of parameters of energy_model is', format(num_params, ','))
# num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
# print('The number of parameters of unet_model is', num_params)

energy = energy.to(device)
# unet = unet.to(device)
      
# use buffer
# buffer = SampleBuffer()
sigmas_np = np.linspace(0.05, args.noise_max, args.batch_size)
sigmas = torch.Tensor(sigmas_np).view((args.batch_size, 1, 1, 1)).to(device)

if args.model_load:
  model_path = '/nas/users/minhyeok/energy_based_model/experiments/model/CIFAR10/2023_05_08/energy.17_26_27.pth'
  energy = torch.load(model_path)

  
# ======================================================================
# Optimizers
# ======================================================================
optim_energy = torch.optim.Adam(energy.parameters(), lr=lr_energy_model, betas=(args.b1, 0.999))
# optim_energy = torch.optim.AdamW(energy.parameters(), lr=lr_energy_model, betas=(args.b1, 0.999))
# optim_unet = torch.optim.AdamW(unet.parameters(), lr=lr_unet, betas=(args.b1, 0.999))
criterion = nn.MSELoss()

# ======================================================================
# Schedulers
# ======================================================================
# ExponentialLR
sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=1, gamma=sched_gamma)
# sched_optim_energy = torch.optim.lr_scheduler.ExponentialLR(optim_energy, gamma=sched_gamma)
# sched_optim_energy = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda = lambda epoch: 0.95 ** epoch)
# sched_optim_energy = torch.optim.lr_scheduler.CosineAnnealingLR(optim_energy, sched_step_size, eta_min=1e-6)
# sched_optim_unet = torch.optim.lr_scheduler.ExponentialLR(optim_unet, gamma=sched_gamma)
# sched_optim_unet = torch.optim.lr_scheduler.LambdaLR(optim_unet, lr_lambda = lambda epoch: 0.95 ** epoch)

# ======================================================================
# Variables for the results 
# ======================================================================

val_loss_energy_model = np.zeros(number_epoch)
val_psnr_ori_mean          = np.zeros(number_epoch)
val_psnr_mean              = np.zeros(number_epoch)
val_psnr_std               = np.zeros(number_epoch)
val_psnr_langevin    = np.zeros(number_epoch)
val_loss_unet_mean         = np.zeros(number_epoch)
exp_weight=0.99
out_avg = None

# ======================================================================
# Training
# ======================================================================

for i in range(number_epoch):
  val_loss_en = list()
  val_ori               = list()
  val_psnr              = list()
  psnr_langevin     = list()
  val_loss_unet         = list()

  start = time.time()

  for j, (image, _) in enumerate(iter(dataloader_train)):
    # -------------------------------------------------------------------
    # Applied degradation
    # -------------------------------------------------------------------
    # if len(buffer) < 1:
    #   buffer.push(image_noise)
    image       = image.to(device)
    image_noise = image + sigmas * torch.randn_like(image)
    random_z = sigmas * torch.randn_like(image)
    for p in energy.parameters():
      p.requires_grad = False
    energy.eval()
    # -------------------------------------------------------------------
    # Y0 generation identity, gaussian distribution, zero
    # -------------------------------------------------------------------
    # unet.train()
    # optim_unet.zero_grad()
    
    # prediction = unet(image_noise)
    # loss_u = criterion(prediction, image)
    
    # loss_u.backward()
    # optim_unet.step()
    
    # sched_optim_unet.step()
    # lr_unet = sched_optim_unet.get_last_lr()[0]

    # prediction = generate_Y0(image_noise, 'id')
    # unet.eval()

    # -------------------------------------------------------------------
    # Yn generation
    # -------------------------------------------------------------------
    # prediction_update = SGLD(image_noise.detach(), energy, number_step_langevin, list_lr_langevin[i])
    noisy = image_noise.clone().detach()
    noisy.requires_grad_()
    for _ in range(args.iter_langevin):
      
      perturb = np.sqrt(list_lr_langevin[i]) * torch.randn_like(noisy)
    
      out_img = - energy(noisy)
      out_img.sum().backward()
      # out_img.requires_grad_(True)
      # out_img.backward()
      noisy.grad.data.clamp_(-0.03, 0.03)
      
      noisy.data = noisy.data - 0.5 * list_lr_langevin[i] * noisy.grad.data + perturb
      
      noisy.grad.detach_()
      noisy.grad.zero_()
      noisy.data.clamp_(-1.0,1.0)
    
    prediction_update = noisy.clone() 
    # -------------------------------------------------------------------
    # grad_x computation
    # -------------------------------------------------------------------
    # prediction_update = noisy.clone()
    # prediction_update = prediction_update.requires_grad_()
    # E = - energy(prediction_update).sum()
    # grad_x = torch.autograd.grad(E, prediction_update, create_graph=True)[0]
    # grad_x = torch.sum(grad_x, (args.batch_size, 1, 1, 1))
    # prediction_update.detach()
    if out_avg is None:
        out_avg = prediction_update.detach()
    else:
        out_avg = out_avg * exp_weight + prediction_update.detach() * (1 - exp_weight)
    # -------------------------------------------------------------------
    # EBM gradient evaluation
    # -------------------------------------------------------------------
    for p in energy.parameters():
      p.requires_grad = True
    
    energy.train()
    optim_energy.zero_grad()

    pos = energy(prediction_update.detach()).mean()
    neg = energy(random_z).mean()
    # grad_x = torch.autograd.grad(neg, prediction_update, create_graph=True)[0]
    loss_energy = neg - pos + L2_w * (neg**2 + pos**2)
    # loss_energy = ((((image - prediction_update)/sigmas/L2_w + grad_x/sigmas)**2)/args.batch_size).sum()
    loss_energy.backward() 
    
    # -------------------------------------------------------------------
    # Update networks
    # -------------------------------------------------------------------
    # clip_grad(energy.parameters(), optim_energy)
    optim_energy.step()

    # -------------------------------------------------------------------
    # Save results for each batch iteration
    # -------------------------------------------------------------------        
    value_psnr_ori      = to_numpy(PSNR(image_noise, image))
    value_psnr_langevin = to_numpy(PSNR(prediction_update, image))
    val_loss_en.append(loss_energy.item())
    val_ori.append(value_psnr_ori)
    psnr_langevin.append(value_psnr_langevin)
      
  # -------------------------------------------------------------------
  # Update schedulers
  # -------------------------------------------------------------------
  lr_energy_model = sched_optim_energy.get_last_lr()[0]
  sched_optim_energy.step()

  # -------------------------------------------------------------------
  # Save results for each epoch
  # -------------------------------------------------------------------
  val_loss_energy_model[i] = np.mean(val_loss_en)
  val_psnr_ori_mean[i]          = np.mean(val_ori)
  val_psnr_langevin[i]     = np.mean(psnr_langevin)

  time_spent = time.time() - start
  log = Self_Energy_log(number_step_langevin) % (i, 
                                                  number_epoch, 
                                                  val_loss_energy_model[i],
                                                  lr_energy_model, 
                                                  val_psnr_ori_mean[i],
                                                  val_psnr_langevin[i],
                                                  time_spent)
  print(f"{name}:\n", log)
  
  fSize = 3
  col = 8
  row = 3
  if i % args.save_plot == 0:
    energy.eval()
    fig, ax = plt.subplots(row, col, figsize = (fSize * 15, fSize * col))
    for k in range(col):
      rand = random.randrange(args.batch_size)
      ax[0][k].set_title('Train (Clean)')
      ax[0][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
      ax[1][k].set_title('Train (Noisy)')
      ax[1][k].imshow(torchvision.utils.make_grid(prediction_update[rand].detach().cpu(), normalize=True).permute(1,2,0))
      ax[2][k].set_title('Train (Langevin)')
      ax[2][k].imshow(torchvision.utils.make_grid(out_avg[rand].detach().cpu(), normalize=True).permute(1,2,0))
    time_ = datetime.datetime.now().strftime('%HH_%MM')

    plt.tight_layout()
    fig.savefig(f'{date_figure}/{name}_log_epoch:{i}_{time_}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
  # TODO: find a condition to break the loop
  if val_psnr_langevin[i] < 3.0 or (i> args.terminate and val_psnr_langevin[i] < val_psnr_langevin[i-10]):
  # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
    print(f'Terminating the run after {i} epochs..')
    break
    
# ======================================================================
# Model evaluation for inferences
# ======================================================================
energy = energy.eval()
# unet = unet.eval()

# ======================================================================
# Training results
# ======================================================================
(image_train, _) = next(iter(dataloader_train))
image_noise_train = gaussian_noise(image_train, args.sigma_noise).to(device)
# image_noise_train = image_train + noise * torch.randn_like(image_train)
# image_noise_train = image_noise_train.to(device)
y_update_train    = SGLD(image_noise_train.detach(), energy, number_step_langevin, list_lr_langevin[-1])

# -------------------------------------------------------------------
# Training performance measuring
# -------------------------------------------------------------------
train_self_data_psnr     = np.zeros(len(dataloader_train))
train_energy_data_psnr = np.zeros(len(dataloader_train))

for j, (image_train_, _) in enumerate(iter(dataloader_train)):
  image_noise_train_ = gaussian_noise(image_train_, args.sigma_noise).to(device)
  image_train_       = image_train_.to(device)
  # image_noise_train_ = image_train_ + noise * torch.randn_like(image_train_).to(device)
  # image_noise_train_ = image_noise_train_.to(device)
  # y_train_           = generate_Y0(image_noise_train_, Y0_type)
  y_update_train_    = SGLD(image_noise_train_.detach(), energy, number_step_langevin, list_lr_langevin[-1])


  train_energy_data_psnr[j] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

# ======================================================================
# Testing results
# ======================================================================
(image_test, _) = next(iter(dataloader_test))
image_test = image_test.to(device)
image_noise_test = image_test + sigmas * torch.randn_like(image_test)
# image_noise_test = gaussian_noise(image_test, args.sigma_noise)
# image_noise_test = image_noise_test.to(device)
# y_test           = generate_Y0(image_noise_test, Y0_type)
y_update_test    = SGLD(image_noise_test.detach(), energy, number_step_langevin, list_lr_langevin[-1])


# ---------------------------------------------------clip_grad-------------
test_self_data_psnr     = np.zeros(len(dataloader_test))
test_energy_data_psnr = np.zeros(len(dataloader_test))

for j, (image_test_, _) in enumerate(iter(dataloader_test)):
  image_test_       = image_test_.to(device)
  # image_noise_test_ = gaussian_noise(image_test_, args.sigma_noise)
  image_noise_test_ = image_test_ + sigmas * torch.randn_like(image_test_).to(device)
  # image_noise_test_ = image_noise_test_.to(device)
  # y_test_           = generate_Y0(image_noise_test_, Y0_type)
  y_update_test_    = SGLD(image_noise_test_.detach(), energy, number_step_langevin, list_lr_langevin[-1])

  test_energy_data_psnr[j] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

time_stamp = datetime.datetime.now().strftime('%H_%M_%S')

file_figure       = os.path.join(date_figure, f'{time_stamp}.png')
file_option       = os.path.join(date_option, f'{time_stamp}.ini')
file_result       = os.path.join(date_result, f'{time_stamp}.csv')
self_file_model   = os.path.join(date_model, f'self.{time_stamp}.pth')
energy_file_model = os.path.join(date_model, f'energy.{time_stamp}.pth')
unet_file_model = os.path.join(date_model, f'unet.{time_stamp}.pth')

# -------------------------------------------------------------------
# Save models
# -------------------------------------------------------------------
torch.save(energy, energy_file_model)
# torch.save(unet, unet_file_model)

# -------------------------------------------------------------------
# Save the options
# -------------------------------------------------------------------
with open(file_option, 'w') as f:
  f.write('{}: {}\n'.format('work directory', dir_work))
  f.write('{}: {}\n'.format('cuda device', cuda_device))
  f.write('{}: {}\n'.format('seed', seed))
  f.write('{}: {}\n'.format('dataset', dir_dataset))
  # f.write('{}: {}\n'.format('Y0 type', Y0_type))
  # f.write('{}: {}\n'.format('image size', im_size))
  # f.write('{}: {}\n'.format('train size', train_size))
  # f.write('{}: {}\n'.format('test size', test_size))
  # f.write('{}: {}\n'.format('use subset', use_subset))
  # f.write('{}: {}\n'.format('use label', use_label))
  f.write('{}: {}\n'.format('number epoch', number_epoch))
  f.write('{}: {}\n'.format('scheduler step size', sched_step_size))
  f.write('{}: {}\n'.format('scheduler gamma', sched_gamma))
  f.write('{}: {}\n'.format('lr energy model', lr_energy_model))
  f.write('{}: {}\n'.format('number step langevin', number_step_langevin))
  f.write('{}: {}\n'.format('regular data', regular_data))
  # f.write('{}: {}\n'.format('use energy weights regularization', use_reg))
  # f.write('{}: {}\n'.format('use energy L2 weights regularization', use_L2_reg))
  f.write('{}: {}\n'.format('energy L2 weights regularization weight', L2_w))
  # f.write('{}: {}\n'.format('add noise', add_noise))
  f.write('{}: {}\n'.format('use unpaired', use_unpaired))
  # f.write('{}: {}\n'.format('sigma noise', sigma_noise))
  # f.write('{}: {}\n'.format('salt and pepper noise', snp_noise))
  # f.write('{}: {}\n'.format('delete square pixels', square_pixels))
  # f.write('{}: {}\n'.format('degradation', degradation))
  f.write('{}: {}\n'.format('in channel', in_channel))
  f.write('{}: {}\n'.format('dim feature', dim_feature))
  f.write('{}: {}\n'.format('dim output', dim_output))
  f.write('{}: {}\n'.format('train energy psnr mean', np.mean(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('train energy psnr std', np.std(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr mean', np.mean(test_energy_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr std', np.std(test_energy_data_psnr)))

f.close()

# -------------------------------------------------------------------
# Save the results
# -------------------------------------------------------------------   
with open(file_result, 'w', newline='') as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(val_loss_energy_model)
  writer.writerow(val_psnr_mean)
  writer.writerow(val_psnr_std)
  writer.writerow(val_psnr_langevin)
  writer.writerow(train_self_data_psnr)
  writer.writerow(train_energy_data_psnr)
  writer.writerow(test_self_data_psnr)
  writer.writerow(test_energy_data_psnr)

f.close()

# -------------------------------------------------------------------
# Save the figures from training
# -------------------------------------------------------------------
nRow  = 7
nCol  = 6

fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('UNet Model')
ax[0][0].plot(val_loss_unet_mean[:i], color='red', label='Loss')
ax[0][0].legend()

ax[0][1].set_title('Energy Model')
ax[0][1].plot(val_loss_energy_model[:i], color='red', label='Loss')
ax[0][1].legend()

ax[0][2].set_title('Train PSNR')
ax[0][2].plot(val_psnr_mean[:i], color='red', label='ID')
ax[0][2].plot(val_psnr_langevin[:i], color='green', label='Langevin')
ax[0][2].legend()

bplot_colors = ['pink', 'lightgreen']

ax[0][3].set_title('Energy Data PSNR')
ax[0][3].yaxis.grid(True)
bplot1 = ax[0][3].boxplot([train_energy_data_psnr, 
                          test_energy_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot1['boxes'], bplot_colors):
  patch.set_facecolor(color)
  
for k in range(nCol):
  rand = random.randrange(args.batch_size)
  ax[1][k].set_title('Train (Input)')
  ax[1][k].imshow(torchvision.utils.make_grid(image_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[2][k].set_title('Train (Artifact)')
  ax[2][k].imshow(torchvision.utils.make_grid(image_noise_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[3][k].set_title('Train (Langevin)')
  ax[3][k].imshow(torchvision.utils.make_grid(y_update_train[rand].detach().cpu(), normalize=True).permute(1,2,0))

  ax[4][k].set_title('Test (Input)')
  ax[4][k].imshow(torchvision.utils.make_grid(image_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[5][k].set_title('Test (Artifact)')
  ax[5][k].imshow(torchvision.utils.make_grid(image_noise_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[6][k].set_title('Test (Langevin)')
  ax[6][k].imshow(torchvision.utils.make_grid(y_update_test[rand].detach().cpu(), normalize=True).permute(1,2,0))

time_1 = datetime.datetime.now().strftime('%HH_%MM')
plt.tight_layout()
fig.savefig(f'{date_figure}/last_{name}_{time_1}.png', bbox_inches='tight', dpi=600)
plt.close(fig)