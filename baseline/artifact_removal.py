import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torchmetrics import PeakSignalNoiseRatio
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime, random
import csv
import os
import argparse
from models import CondEnergyModel, EnergyModel, UNet
from sn_gan import Discriminator
from utils import UNet_Energy_log, to_numpy, init_weights, Self_Energy_log, Custom_Dataset, SGLD
from utils import gaussian_noise, sp_noise, delete_square, generate_Y0, clip_grad, SampleBuffer

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='first')
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--in_channel", type=int, default=1)
parser.add_argument("--img_size", type=int, default=32)
parser.add_argument("--dim_feature", type=int, default=64)

parser.add_argument("--lr_unet", type=float, default=0.0001)
parser.add_argument("--lr_energy_model", type=float, default=0.00005)
parser.add_argument("--lr_langevin_min", type=float, default=0.000005)
parser.add_argument("--lr_langevin_max", type=float, default=0.00001)
parser.add_argument("--number_step_langevin", type=int, default=6)

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--save_plot", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("--b1", type=float, default=0.2, help="adam: decay of first order momentum of gradient")

parser.add_argument("--use_unpaired", action="store_true", default=False)
parser.add_argument("--regular_data", type=float, default=0.0)
parser.add_argument("--init_noise_decay", type=float, default=1.0)
# parser.add_argument("--use_gp", action="store_true", default=True)
# parser.add_argument("--use_energy_reg", action="store_true", default=False)
parser.add_argument("--energy_reg_weight", type=float, default=0.0005)
parser.add_argument("--use_en_L2_reg", action="store_true", default=False)
parser.add_argument("--energy_L2_reg_weight", type=float, default=0.001)

parser.add_argument("--model_param_load", type=bool, default=False)

args = parser.parse_args()

# ======================================================================
# Options
# ======================================================================
# curr_dir_path         = os.path.join(os.path.dirname(os.path.abspath('')), '..', 'experiments')
dir_work              = '/nas/users/minhyeok/energy_based_model/experiments'
name                  = args.name
cuda_device           = 0
NGPU                  = torch.cuda.device_count()
device                = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
batch_size            = args.batch_size
number_epoch          = args.epochs
use_energy_sched      = True
sched_step_size       = number_epoch // 10
sched_gamma           = 0.97
lr_energy_model       = args.lr_energy_model
lr_unet               = args.lr_unet
lr_langevin_max       = args.lr_langevin_max
lr_langevin_min       = args.lr_langevin_min
number_step_langevin  = args.number_step_langevin
regular_data          = args.regular_data
init_noise_decay      = args.init_noise_decay
# use_reg               = args.use_energy_reg
reg_weight            = args.energy_reg_weight
# use_L2_reg            = args.use_energy_L2_reg
L2_reg_weight         = args.energy_L2_reg_weight
use_unpaired          = args.use_unpaired
in_channel            = args.in_channel
dim_feature           = args.dim_feature
dim_output            = 1
# use_gp                = args.use_gp
seed                  = args.seed
list_lr_langevin      = np.linspace(lr_langevin_max, lr_langevin_min, num=number_epoch, endpoint=True)
pl.seed_everything(seed)

# ======================================================================
# Dataset 
# ======================================================================
# dir_data    = os.path.join(dir_work, 'data')
# dir_data = '/hdd1/dataset/'
dir_data    = '/nas/dataset/users/minhyeok/recon'
dir_dataset = 'StreakArtifact'
scale_range = [-1, 1] # [0,1]

# ======================================================================
# Path for the results
# ======================================================================
dir_figure = os.path.join(dir_work, 'figure')
dir_option = os.path.join(dir_work, 'option')
dir_result = os.path.join(dir_work, 'result')
dir_model  = os.path.join(dir_work, 'model')
now        = datetime.datetime.now()
date_stamp = now.strftime('%Y_%m_%d')
time_stamp = now.strftime('%H_%M_%S')

path_figure = os.path.join(dir_figure, dir_dataset)
path_option = os.path.join(dir_option, dir_dataset)
path_result = os.path.join(dir_result, dir_dataset)
path_model  = os.path.join(dir_model, dir_dataset)

date_figure = os.path.join(path_figure, date_stamp)
date_option = os.path.join(path_option, date_stamp)
date_result = os.path.join(path_result, date_stamp)
date_model  = os.path.join(path_model, date_stamp)

file_figure       = os.path.join(date_figure, f'{time_stamp}.png')
file_option       = os.path.join(date_option, f'{time_stamp}.ini')
file_result       = os.path.join(date_result, f'{time_stamp}.csv')
self_file_model   = os.path.join(date_model, f'self.{time_stamp}.pth')
energy_file_model = os.path.join(date_model, f'energy.{time_stamp}.pth')
unet_file_model = os.path.join(date_model, f'unet.{time_stamp}.pth')
  
if not os.path.exists(dir_figure):  os.makedirs(dir_figure)

if not os.path.exists(dir_option):  os.makedirs(dir_option)

if not os.path.exists(dir_result):  os.makedirs(dir_result)

if not os.path.exists(dir_model):  os.makedirs(dir_model)

if not os.path.exists(path_figure):  os.makedirs(path_figure)

if not os.path.exists(path_option):  os.makedirs(path_option)

if not os.path.exists(path_result):  os.makedirs(path_result)

if not os.path.exists(path_model):  os.makedirs(path_model)

if not os.path.exists(date_figure):  os.makedirs(date_figure)

if not os.path.exists(date_option):  os.makedirs(date_option)

if not os.path.exists(date_result):  os.makedirs(date_result)

if not os.path.exists(date_model):  os.makedirs(date_model)
  

# ======================================================================
# Dataloaders 
# ======================================================================
# dataloader_train, dataloader_test, _, vis_im_transform = get_dataloaders(dir_data, dir_dataset, im_size, 
#                                                                          in_channel, batch_size, train_size, 
#                                                                          train_ratio, test_size, 
#                                                                          use_subset, use_label, 
#                                                                          scale_range, use_unpaired, num_workers=NGPU)

transforms_train = transforms.Compose([
    torchvision.transforms.Resize(args.img_size),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))])
train_set = Custom_Dataset(dir_data, transform=transforms_train)
test_set = Custom_Dataset(dir_data, transform=transforms_train, train=False)
dataloader_train = DataLoader(dataset=train_set, batch_size=args.batch_size, drop_last=True, shuffle=True)
dataloader_test = DataLoader(dataset=test_set, batch_size=args.batch_size, drop_last=True, shuffle=False)

# ======================================================================
# Evaluation
# ======================================================================
PSNR = PeakSignalNoiseRatio().to(device)

# ======================================================================
# Instantiations
# ======================================================================

energy = Discriminator(in_channel, dim_feature, activation='swish')

unet = UNet(in_channel, dim_feature, in_channel)
num_params = sum(p.numel() for p in energy.parameters() if p.requires_grad)
print('The number of parameters of energy_model is', num_params)
num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print('The number of parameters of unet_model is', num_params)

if NGPU > 1:
  energy = torch.nn.DataParallel(energy, device_ids=list(range(NGPU)))
  energy = energy.module.to(device)
else:
  energy = energy.to(device)
  unet = unet.to(device)
      
# use buffer
# buffer = SampleBuffer()

# ======================================================================
# Weights initializations
# ======================================================================
# TODO: implement weights initialization method
def init_weights(m):
  """
  Applies initial weights to certain layers in a model: convolutional and linear
  The weights are taken from a normal distribution 
  with mean = 0, std dev = 0.02.
  :param m: A module or layer in a network    
  """
  # classname will be something like:
  # `Conv`, `BatchNorm2d`, `Linear`, etc.
  classname = m.__class__.__name__
  isConvolution = classname.find('Conv') != -1
  isLinear = classname.find('Linear') != -1
  isNorm = classname.find('BatchNorm') != -1
  if (hasattr(m, 'weight') and isConvolution or isLinear):
    nn.init.kaiming_uniform_(m.weight.data)
    # nn.init.xavier_uniform_(m.weight.data)
    # nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif (hasattr(m, 'bias') and isConvolution or isLinear):
    nn.init.constant_(m.weight.data, 0)

if args.model_param_load:
  model_path = '/nas/users/minhyeok/energy_based_model/experiments/model/CIFAR10/2023_05_08/energy.17_26_27.pth'
  energy = torch.load(model_path)
else:
  # energy.apply(init_weights)
  unet.apply(init_weights)
  
# ======================================================================
# Optimizers
# ======================================================================
# optim_energy = torch.optim.Adam(energy.parameters(), lr=lr_energy_model, betas=(args.b1, 0.999))
optim_energy = torch.optim.AdamW(energy.parameters(), lr=lr_energy_model, betas=(args.b1, 0.999))
optim_unet = torch.optim.AdamW(unet.parameters(), lr=lr_unet, betas=(args.b1, 0.999))
criterion = nn.MSELoss(reduce='sum')

# ======================================================================
# Schedulers
# ======================================================================
# ExponentialLR
# sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=sched_step_size, gamma=sched_gamma)
# sched_optim_energy = torch.optim.lr_scheduler.ExponentialLR(optim_energy, gamma=sched_gamma)
# sched_optim_energy = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda = lambda epoch: 0.95 ** epoch)

# sched_optim_unet = torch.optim.lr_scheduler.ExponentialLR(optim_unet, gamma=sched_gamma)
# sched_optim_unet = torch.optim.lr_scheduler.LambdaLR(optim_unet, lr_lambda = lambda epoch: 0.95 ** epoch)

# ======================================================================
# Variables for the results 
# ======================================================================

val_loss_energy_model_mean = np.zeros(number_epoch)
val_loss_energy_model_std  = np.zeros(number_epoch)
val_psnr_ori_mean          = np.zeros(number_epoch)
val_psnr_mean              = np.zeros(number_epoch)
val_psnr_std               = np.zeros(number_epoch)
val_psnr_langevin_mean     = np.zeros(number_epoch)
val_psnr_langevin_std      = np.zeros(number_epoch)
val_loss_unet_mean         = np.zeros(number_epoch)


# ======================================================================
# Training
# ======================================================================

for i in range(number_epoch):
  val_loss_energy_model = list()
  val_ori               = list()
  val_psnr              = list()
  val_psnr_langevin     = list()
  val_loss_unet         = list()

  for j, (image, image_noise) in enumerate(iter(dataloader_train)):
    # -------------------------------------------------------------------
    # Applied degradation
    # -------------------------------------------------------------------
    # if len(buffer) < 1:
    #   buffer.push(image_noise)
    image       = image.to(device)
    image_noise = image_noise.to(device)
    energy.eval()
    for p in energy.parameters():
      p.requires_grad = False
    # -------------------------------------------------------------------
    # Y0 generation identity, gaussian distribution, zero
    # -------------------------------------------------------------------
    unet.train()
    optim_unet.zero_grad()
    
    prediction = unet(image_noise)
    # loss_u = unet.compute_loss(prediction, image)
    loss_u = criterion(prediction, image)
    
    loss_u.backward()
    optim_unet.step()
    
    # sched_optim_unet.step()
    # lr_unet = sched_optim_unet.get_last_lr()[0]

    # prediction = generate_Y0(image_noise, 'id')
    unet.eval()

    # -------------------------------------------------------------------
    # Yn generation
    # -------------------------------------------------------------------
    prediction_update = SGLD(prediction.detach(), energy, number_step_langevin, list_lr_langevin[i])

    # -------------------------------------------------------------------
    # EBM gradient evaluation
    # -------------------------------------------------------------------
    for p in energy.parameters():
      p.requires_grad = True
    energy = energy.train()
    optim_energy.zero_grad()

    neg = energy(prediction_update.detach())
    pos = energy(image)
    loss_energy = neg.mean() - pos.mean()
    loss_energy.backward() 

    value_psnr_ori      = to_numpy(PSNR(image_noise, image))
    value_psnr          = to_numpy(PSNR(prediction, image))
    value_psnr_langevin = to_numpy(PSNR(prediction_update, image))

    # -------------------------------------------------------------------
    # Update networks
    # -------------------------------------------------------------------
    # clip_grad(energy.parameters(), optim_energy)
    optim_energy.step()

    # -------------------------------------------------------------------
    # Save results for each batch iteration
    # -------------------------------------------------------------------        
    val_loss_energy_model.append(loss_energy.item())
    val_loss_unet.append(loss_u.item())
    val_ori.append(value_psnr_ori)
    val_psnr.append(value_psnr)
    val_psnr_langevin.append(value_psnr_langevin)
      
  # -------------------------------------------------------------------
  # Update schedulers
  # -------------------------------------------------------------------
  # lr_energy_model = sched_optim_energy.get_last_lr()[0]

  # if use_energy_sched:
    # sched_optim_energy.step()

  # -------------------------------------------------------------------
  # Save results for each epoch
  # -------------------------------------------------------------------
  val_loss_energy_model_mean[i] = np.mean(val_loss_energy_model)
  val_loss_energy_model_std[i]  = np.std(val_loss_energy_model)
  val_loss_unet_mean[i]         = np.mean(val_loss_unet)
  val_psnr_ori_mean[i]          = np.mean(val_ori)
  val_psnr_mean[i]              = np.mean(val_psnr)
  val_psnr_std[i]               = np.std(val_psnr)
  val_psnr_langevin_mean[i]     = np.mean(val_psnr_langevin)
  val_psnr_langevin_std[i]      = np.std(val_psnr_langevin)

  log = UNet_Energy_log(number_step_langevin) % (i, 
                                                  number_epoch, 
                                                  val_loss_unet_mean[i],
                                                  val_loss_energy_model_mean[i],
                                                  lr_unet, 
                                                  lr_energy_model, 
                                                  val_psnr_ori_mean[i],
                                                  val_psnr_mean[i], 
                                                  val_psnr_langevin_mean[i])
  print(f"{name}:\n", log)
  
  
  if i % args.save_plot == 0:
    energy.eval()
    fig, ax = plt.subplots(4, 10, figsize = (3 * 15, 3 * 10))
    for k in range(10):
      rand = random.randrange(args.batch_size)
      ax[0][k].set_title('Train (Input)')
      ax[0][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
      ax[1][k].set_title('Train (Artifact)')
      ax[1][k].imshow(torchvision.utils.make_grid(image_noise[rand].detach().cpu(), normalize=True).permute(1,2,0))
      ax[2][k].set_title('Train (UNet)')
      ax[2][k].imshow(torchvision.utils.make_grid(prediction[rand].detach().cpu(), normalize=True).permute(1,2,0))
      ax[3][k].set_title('Train (Langevin)')
      ax[3][k].imshow(torchvision.utils.make_grid(prediction_update[rand].detach().cpu(), normalize=True).permute(1,2,0))
    time_ = datetime.datetime.now().strftime('%HH_%MM')

    plt.tight_layout()
    fig.savefig(f'{date_figure}/{name}_log_epoch:{i}_{time_}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
  # TODO: find a condition to break the loop
  if i>20 and val_psnr_langevin_mean[i] < val_psnr_langevin_mean[i-5]:
  # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
    print(f'Terminating the run after {i} epochs..')
    break
    
# ======================================================================
# Model evaluation for inferences
# ======================================================================
energy = energy.eval()
unet = unet.eval()

# ======================================================================
# Training results
# ======================================================================
(image_train, image_noise_train) = next(iter(dataloader_train))
# image_train       = image_train.to(device)
image_noise_train = image_noise_train.to(device)
y_train           = unet(image_noise_train)
# y_train           = generate_Y0(image_noise_train, Y0_type)
y_update_train    = SGLD(y_train.detach(), energy, number_step_langevin, list_lr_langevin[-1])
# y_update_train    = energy.update_prediction_langevin( 
#                                                       y_train.detach(), 
#                                                       number_step_langevin, 
#                                                       list_lr_langevin[-1], 
#                                                       regular_data, 
#                                                       add_noise, 
#                                                       init_noise_decay)

# image_train       = to_numpy(permute(image_train)).squeeze()
# image_noise_train = to_numpy(permute(image_noise_train)).squeeze()
# y_train           = to_numpy(permute(y_train)).squeeze()
# y_update_train    = to_numpy(permute(y_update_train)).squeeze()

# -------------------------------------------------------------------
# Training performance measuring
# -------------------------------------------------------------------
train_self_data_psnr     = np.zeros(len(dataloader_train))
train_energy_data_psnr = np.zeros(len(dataloader_train))

for j, (image_train_, image_noise_train_) in enumerate(iter(dataloader_train)):
  image_train_       = image_train_.to(device)
  image_noise_train_ = image_noise_train_.to(device)
  y_train_           = unet(image_noise_train_)
  # y_train_           = generate_Y0(image_noise_train_, Y0_type)
  y_update_train_    = SGLD(y_train_.detach(), energy, number_step_langevin, list_lr_langevin[-1])
  # y_update_train_    = energy.update_prediction_langevin(
  #                                                        y_train_.detach(), 
  #                                                        number_step_langevin, 
  #                                                        list_lr_langevin[-1], 
  #                                                        regular_data, 
  #                                                        add_noise, 
  #                                                        init_noise_decay)

  train_self_data_psnr[j]     = to_numpy(PSNR(y_train_, image_train_)).mean()
  train_energy_data_psnr[j] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

# ======================================================================
# Testing results
# ======================================================================
(image_test, image_noise_test) = next(iter(dataloader_test))
# image_test       = image_test.to(device)
image_noise_test = image_noise_test.to(device)
y_test           = unet(image_noise_test)
# y_test           = generate_Y0(image_noise_test, Y0_type)
y_update_test    = SGLD(y_test.detach(), energy, number_step_langevin, list_lr_langevin[-1])
# y_update_test    = energy.update_prediction_langevin(
#                                                      y_test.detach(), 
#                                                      number_step_langevin, 
#                                                      list_lr_langevin[-1], 
#                                                      regular_data, 
#                                                      add_noise, 
#                                                      init_noise_decay)

# image_test       = to_numpy(permute(image_test)).squeeze()
# image_noise_test = to_numpy(permute(image_noise_test)).squeeze()
# y_test           = to_numpy(permute(y_test)).squeeze()
# y_update_test    = to_numpy(permute(y_update_test)).squeeze()

# ---------------------------------------------------clip_grad-------------
test_self_data_psnr     = np.zeros(len(dataloader_test))
test_energy_data_psnr = np.zeros(len(dataloader_test))

for j, (image_test_, image_noise_test_) in enumerate(iter(dataloader_test)):
  image_test_       = image_test_.to(device)
  image_noise_test_ = image_noise_test_.to(device)
  y_test_           = unet(image_noise_test_)
  # y_test_           = generate_Y0(image_noise_test_, Y0_type)
  y_update_test_    = SGLD(y_test_.detach(), energy, number_step_langevin, list_lr_langevin[-1])
  # y_update_test_    = energy.update_prediction_langevin(
  #                                                       y_test_.detach(), 
  #                                                       number_step_langevin, 
  #                                                       list_lr_langevin[-1], 
  #                                                       regular_data, 
  #                                                       add_noise, 
  #                                                       init_noise_decay)

  test_self_data_psnr[j]     = to_numpy(PSNR(y_test_, image_test_)).mean()
  test_energy_data_psnr[j] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

# -------------------------------------------------------------------
# Save models
# -------------------------------------------------------------------
torch.save(energy, energy_file_model)
torch.save(unet, unet_file_model)

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
  f.write('{}: {}\n'.format('scale range', scale_range))
  # f.write('{}: {}\n'.format('train size', train_size))
  # f.write('{}: {}\n'.format('test size', test_size))
  # f.write('{}: {}\n'.format('use subset', use_subset))
  # f.write('{}: {}\n'.format('use label', use_label))
  f.write('{}: {}\n'.format('batch size', batch_size))
  f.write('{}: {}\n'.format('number epoch', number_epoch))
  f.write('{}: {}\n'.format('use energy scheduler', use_energy_sched))
  f.write('{}: {}\n'.format('scheduler step size', sched_step_size))
  f.write('{}: {}\n'.format('scheduler gamma', sched_gamma))
  f.write('{}: {}\n'.format('lr energy model', lr_energy_model))
  f.write('{}: {}\n'.format('lr langevin max', lr_langevin_max))
  f.write('{}: {}\n'.format('lr langevin min', lr_langevin_min))
  f.write('{}: {}\n'.format('number step langevin', number_step_langevin))
  f.write('{}: {}\n'.format('regular data', regular_data))
  f.write('{}: {}\n'.format('langevin noise decay factor', init_noise_decay))
  # f.write('{}: {}\n'.format('use energy weights regularization', use_reg))
  f.write('{}: {}\n'.format('energy weights regularization weight', reg_weight))
  # f.write('{}: {}\n'.format('use energy L2 weights regularization', use_L2_reg))
  f.write('{}: {}\n'.format('energy L2 weights regularization weight', L2_reg_weight))
  # f.write('{}: {}\n'.format('add noise', add_noise))
  f.write('{}: {}\n'.format('use unpaired', use_unpaired))
  # f.write('{}: {}\n'.format('sigma noise', sigma_noise))
  # f.write('{}: {}\n'.format('salt and pepper noise', snp_noise))
  # f.write('{}: {}\n'.format('delete square pixels', square_pixels))
  # f.write('{}: {}\n'.format('degradation', degradation))
  f.write('{}: {}\n'.format('in channel', in_channel))
  f.write('{}: {}\n'.format('dim feature', dim_feature))
  f.write('{}: {}\n'.format('dim output', dim_output))
  f.write('{}: {}\n'.format('train id psnr mean', np.mean(train_self_data_psnr)))
  f.write('{}: {}\n'.format('train id psnr std', np.std(train_self_data_psnr)))
  f.write('{}: {}\n'.format('train energy psnr mean', np.mean(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('train energy psnr std', np.std(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('test id psnr mean', np.mean(test_self_data_psnr)))
  f.write('{}: {}\n'.format('test id psnr std', np.std(test_self_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr mean', np.mean(test_energy_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr std', np.std(test_energy_data_psnr)))

f.close()

# -------------------------------------------------------------------
# Save the results
# -------------------------------------------------------------------   
with open(file_result, 'w', newline='') as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(val_loss_energy_model_mean)
  writer.writerow(val_loss_energy_model_std)
  writer.writerow(val_psnr_mean)
  writer.writerow(val_psnr_std)
  writer.writerow(val_psnr_langevin_mean)
  writer.writerow(val_psnr_langevin_std)
  writer.writerow(train_self_data_psnr)
  writer.writerow(train_energy_data_psnr)
  writer.writerow(test_self_data_psnr)
  writer.writerow(test_energy_data_psnr)

f.close()

# -------------------------------------------------------------------
# Save the figures from training
# -------------------------------------------------------------------
nRow  = 9
nCol  = 6
fSize = 3

fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('UNet Model')
ax[0][0].plot(val_loss_unet_mean[:i], color='red', label='Loss')
ax[0][0].legend()

ax[0][1].set_title('Energy Model')
ax[0][1].plot(val_loss_energy_model_mean[:i], color='red', label='Loss')
ax[0][1].legend()

ax[0][2].set_title('Train PSNR')
ax[0][2].plot(val_psnr_mean[:i], color='red', label='ID')
ax[0][2].plot(val_psnr_langevin_mean[:i], color='green', label='Langevin')
ax[0][2].legend()

bplot_colors = ['pink', 'lightgreen']

ax[0][3].set_title('UNet Data PSNR')
ax[0][3].yaxis.grid(True)
bplot0 = ax[0][3].boxplot([train_self_data_psnr, 
                          test_self_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot0['boxes'], bplot_colors):
  patch.set_facecolor(color)

ax[0][4].set_title('Energy Data PSNR')
ax[0][4].yaxis.grid(True)
bplot1 = ax[0][4].boxplot([train_energy_data_psnr, 
                          test_energy_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot1['boxes'], bplot_colors):
  patch.set_facecolor(color)
  
for k in range(6):
  rand = random.randrange(args.batch_size)
  ax[1][k].set_title('Train (Input)')
  ax[1][k].imshow(torchvision.utils.make_grid(image_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[2][k].set_title('Train (Artifact)')
  ax[2][k].imshow(torchvision.utils.make_grid(image_noise_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[3][k].set_title('Train (UNet)')
  ax[3][k].imshow(torchvision.utils.make_grid(y_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[4][k].set_title('Train (Langevin)')
  ax[4][k].imshow(torchvision.utils.make_grid(y_update_train[rand].detach().cpu(), normalize=True).permute(1,2,0))

  ax[5][k].set_title('Test (Input)')
  ax[5][k].imshow(torchvision.utils.make_grid(image_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[6][k].set_title('Test (Artifact)')
  ax[6][k].imshow(torchvision.utils.make_grid(image_noise_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[7][k].set_title('Test (UNet)')
  ax[7][k].imshow(torchvision.utils.make_grid(y_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
  
  ax[8][k].set_title('Test (Langevin)')
  ax[8][k].imshow(torchvision.utils.make_grid(y_update_test[rand].detach().cpu(), normalize=True).permute(1,2,0))

time_1 = datetime.datetime.now().strftime('%HH_%MM')
plt.tight_layout()
fig.savefig(f'{date_figure}/last_{name}_{time_1}.png', bbox_inches='tight', dpi=600)
plt.close(fig)