import torch
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import csv
import os
import argparse
from models import UNet, CondEnergyModel
from utils import permute, to_numpy, init_weights, UNet_Energy_log, get_dataloaders
from utils import gaussian_noise, sp_noise, delete_square

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--train_size", type=int, default=60000)
parser.add_argument("--test_size", type=int, default=10000)
parser.add_argument("--in_channel", type=int, default=1)
parser.add_argument("--dim_feature", type=int, default=32)
parser.add_argument("--gaussian_noise", type=float, default=0.75)
parser.add_argument("--sp_noise", type=float, default=0.1)
parser.add_argument("--square_pixels", type=int, default=20)
parser.add_argument("--degradation", type=str, default='gaussian_noise')
parser.add_argument("--lr_unet", type=float, default=0.005)
parser.add_argument("--lr_energy_model", type=float, default=0.001)
parser.add_argument("--lr_langevin_max", type=float, default=0.001)
parser.add_argument("--lr_langevin_min", type=float, default=0.001)
parser.add_argument("--number_step_langevin", type=int, default=10)
parser.add_argument("--langevin_steps_factor", type=int, default=1)
parser.add_argument("--use_energy_sched", action="store_true", default=False)
parser.add_argument("--use_subset", action="store_true", default=False)
parser.add_argument("--use_label", type=int, default=4)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--train_ratio", type=float, default=1.0)
parser.add_argument("--use_unpaired", action="store_true", default=False)
parser.add_argument("--regular_data", type=float, default=0.0)
parser.add_argument("--noise_decay", type=float, default=1.0)
parser.add_argument("--use_energy_reg", action="store_true", default=False)
parser.add_argument("--energy_reg_weight", type=float, default=0.0)
parser.add_argument("--use_gp", type=bool, default=False)
parser.add_argument("--gp_weight", type=float, default=0.0)
parser.add_argument("--use_energy_L2_reg", action="store_true", default=False)
parser.add_argument("--energy_L2_reg_weight", type=float, default=0.0)
args = parser.parse_args()


# ======================================================================
# Options
# ======================================================================
curr_dir_path         = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
name                  = args.name
dir_work              = os.path.join(curr_dir_path, name)
cuda_device           = 0
device                = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
batch_size            = args.batch_size
number_epoch          = args.epochs
lr_unet               = args.lr_unet
use_unet_sched        = False
use_energy_sched      = args.use_energy_sched
sched_step_size       = 10
sched_gamma           = 0.93
lr_energy_model       = args.lr_energy_model
lr_langevin_max       = args.lr_langevin_max
lr_langevin_min       = args.lr_langevin_min
number_step_langevin  = args.number_step_langevin
langevin_steps_factor = args.langevin_steps_factor
regular_data          = args.regular_data
noise_decay           = args.noise_decay
use_reg               = args.use_energy_reg
reg_weight            = args.energy_reg_weight
use_gp                = args.use_gp
gp_weight             = args.gp_weight
use_L2_reg            = args.use_energy_L2_reg
L2_reg_weight         = args.energy_L2_reg_weight
add_noise             = True
use_unpaired          = args.use_unpaired
in_channel            = args.in_channel
dim_feature           = args.dim_feature
dim_output            = 1
degradation           = args.degradation
sigma_noise           = args.gaussian_noise
snp_noise             = args.sp_noise
square_pixels         = args.square_pixels
seed                  = 0
list_lr_langevin      = np.linspace(lr_langevin_max, lr_langevin_min, num=number_epoch, endpoint=True)
pl.seed_everything(0)

# ======================================================================
# Apply degradation
# ======================================================================
if degradation == 'gaussian_noise':
  degradation_func = lambda im: gaussian_noise(im, sigma_noise)

elif degradation == 'sp_noise':
  degradation_func = lambda im: sp_noise(im, snp_noise)

elif degradation == 'delete_square':
  degradation_func = lambda im: delete_square(im, square_pixels)

# ======================================================================
# Dataset 
# ======================================================================
dir_data    = os.path.join(dir_work, 'data')
dir_dataset = args.dataset
im_size     = 32
scale_range = [-1, 1]
train_size  = args.train_size
test_size   = args.test_size
use_subset  = args.use_subset
use_label   = args.use_label
train_ratio = args.train_ratio

# ======================================================================
# Dataloaders 
# ======================================================================
dataloader_train, dataloader_test, out_activation, vis_im_transform = get_dataloaders(dir_data, dir_dataset, im_size, 
                                                                                      in_channel, batch_size, train_size, 
                                                                                      train_ratio, test_size, 
                                                                                      use_subset, use_label, 
                                                                                      scale_range, use_unpaired)

# ======================================================================
# Evaluation
# ======================================================================
PSNR = PeakSignalNoiseRatio().to(device)

# ======================================================================
# Instantiations
# ======================================================================
unet   = UNet(in_channel=in_channel, dim_feature=dim_feature, out_channel=in_channel, bilinear=True, out_activation=out_activation).to(device)
energy = CondEnergyModel(in_channel * 2, dim_feature, dim_output, scale_range=scale_range,
                                                                  use_gp=use_gp, 
                                                                  gp_weight=gp_weight, 
                                                                  use_reg=use_reg, 
                                                                  reg_weight=reg_weight, 
                                                                  use_L2_reg=use_L2_reg, 
                                                                  L2_reg_weight=L2_reg_weight).to(device)
                                                                                                       
# ======================================================================
# Weights initializations
# ======================================================================
# unet.apply(init_weights)
# TODO: implement weights initialization method
# energy.apply(init_weights)

# ======================================================================
# Optimizers
# ======================================================================
optim_unet   = torch.optim.Adam(unet.parameters(), lr=lr_unet)
optim_energy = torch.optim.Adam(energy.parameters(), lr=lr_energy_model)

# ======================================================================
# Schedulers
# ======================================================================
# ExponentialLR
sched_optim_unet   = torch.optim.lr_scheduler.StepLR(optim_unet, step_size=sched_step_size, gamma=sched_gamma)
sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=sched_step_size, gamma=sched_gamma)

# ======================================================================
# Variables for the results 
# ======================================================================
val_loss_unet_mean         = np.zeros(number_epoch)
val_loss_unet_std          = np.zeros(number_epoch)
val_loss_energy_model_mean = np.zeros(number_epoch)
val_loss_energy_model_std  = np.zeros(number_epoch)
val_psnr_mean              = np.zeros(number_epoch)
val_psnr_std               = np.zeros(number_epoch)
val_psnr_langevin_mean     = np.zeros(number_epoch)
val_psnr_langevin_std      = np.zeros(number_epoch)

# ======================================================================
# Training
# ======================================================================
unet   = unet.train()
energy = energy.train()

for i in range(number_epoch):
  val_loss_unet         = list()
  val_loss_energy_model = list()
  val_psnr              = list()
  val_psnr_langevin     = list()
  
  for j, (image, _) in enumerate(iter(dataloader_train)):
    # -------------------------------------------------------------------
    # Applied degradation
    # -------------------------------------------------------------------
    if use_unpaired:
      image, image2 = torch.split(image, batch_size)
      image2_noise  = degradation_func(image2)
      image2        = image2.to(device)
      image2_noise  = image2_noise.to(device)

    else:
      image_noise  = degradation_func(image)

    image       = image.to(device)
    image_noise = image_noise.to(device)

    # -------------------------------------------------------------------
    # Y0 generation
    # -------------------------------------------------------------------
    prediction = unet(image_noise)

    # -------------------------------------------------------------------
    # Yn generation
    # -------------------------------------------------------------------
    prediction_update = energy.update_prediction_langevin(image_noise, 
                                                          prediction.detach(), 
                                                          number_step_langevin,  
                                                          list_lr_langevin[i], 
                                                          regular_data, 
                                                          add_noise, 
                                                          noise_decay)
    # -------------------------------------------------------------------
    # EBM gradient evaluation
    # -------------------------------------------------------------------
    optim_energy.zero_grad()
    if use_unpaired:
      loss_energy = energy.compute_loss(image2_noise, image2, image_noise, prediction_update.detach())
    
    else:
      loss_energy = energy.compute_loss(image_noise, image, image_noise, prediction_update.detach())
    loss_energy.backward()

    # -------------------------------------------------------------------
    # UNet gradient evaluation
    # -------------------------------------------------------------------
    optim_unet.zero_grad()
    loss_unet = unet.compute_loss(prediction, prediction_update.detach())
    loss_unet.backward()
    
    value_psnr          = to_numpy(PSNR(prediction, image))
    value_psnr_langevin = to_numpy(PSNR(prediction_update, image))

    # -------------------------------------------------------------------
    # Update networks
    # -------------------------------------------------------------------
    optim_unet.step()
    optim_energy.step()
    
    # -------------------------------------------------------------------
    # Save results for each batch iteration
    # -------------------------------------------------------------------        
    val_loss_unet.append(loss_unet.item())
    val_loss_energy_model.append(loss_energy.item())
    val_psnr.append(value_psnr)
    val_psnr_langevin.append(value_psnr_langevin)

  # -------------------------------------------------------------------
  # Update schedulers
  # -------------------------------------------------------------------
  lr_unet         = sched_optim_unet.get_last_lr()[0]
  lr_energy_model = sched_optim_energy.get_last_lr()[0]

  if use_unet_sched:
    sched_optim_unet.step()
  if use_energy_sched:
    sched_optim_energy.step()

  # -------------------------------------------------------------------
  # Save results for each epoch
  # -------------------------------------------------------------------        
  val_loss_unet_mean[i]         = np.mean(val_loss_unet)
  val_loss_unet_std[i]          = np.std(val_loss_unet)
  val_loss_energy_model_mean[i] = np.mean(val_loss_energy_model)
  val_loss_energy_model_std[i]  = np.std(val_loss_energy_model)
  val_psnr_mean[i]              = np.mean(val_psnr)
  val_psnr_std[i]               = np.std(val_psnr)
  val_psnr_langevin_mean[i]     = np.mean(val_psnr_langevin)
  val_psnr_langevin_std[i]      = np.std(val_psnr_langevin)

  log = UNet_Energy_log(number_step_langevin) % (i, 
                                                 number_epoch, 
                                                 val_loss_unet_mean[i], 
                                                 lr_unet, 
                                                 val_loss_energy_model_mean[i], 
                                                 lr_energy_model, 
                                                 val_psnr_mean[i], 
                                                 val_psnr_langevin_mean[i])
  print(f"{name}:\n", log, flush=True)
  
  # TODO: find a condition to break the loop
  # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
  #   print(f'Terminating the run after {i} epochs..')
  #   break

# ======================================================================
# Model evaluation for inferences
# ======================================================================
unet   = unet.eval()
energy = energy.eval()

# ======================================================================
# Training results
# ======================================================================
(image_train, _) = next(iter(dataloader_train))
image_noise_train = degradation_func(image_train)
image_train       = image_train.to(device)
image_noise_train = image_noise_train.to(device)
y_train           = unet(image_noise_train)
y_update_train    = energy.update_prediction_langevin(image_noise_train, 
                                                      y_train.detach(), 
                                                      number_step_langevin * langevin_steps_factor, 
                                                      list_lr_langevin[-1], 
                                                      regular_data, 
                                                      add_noise, 
                                                      noise_decay)

image_train       = to_numpy(permute(image_train)).squeeze()
image_noise_train = to_numpy(permute(image_noise_train)).squeeze()
y_train           = to_numpy(permute(y_train)).squeeze()
y_update_train    = to_numpy(permute(y_update_train)).squeeze()

# -------------------------------------------------------------------
# Training performance measuring
# -------------------------------------------------------------------
train_unet_data_psnr   = np.zeros(len(dataloader_train))
train_energy_data_psnr = np.zeros(len(dataloader_train))

for i, (image_train_, _) in enumerate(iter(dataloader_train)):
  image_noise_train_ = degradation_func(image_train_)
  image_train_       = image_train_.to(device)
  image_noise_train_ = image_noise_train_.to(device)
  y_train_           = unet(image_noise_train_)
  y_update_train_    = energy.update_prediction_langevin(image_noise_train_, 
                                                         y_train_.detach(), 
                                                         number_step_langevin * langevin_steps_factor, 
                                                         list_lr_langevin[-1], 
                                                         regular_data, 
                                                         add_noise, 
                                                         noise_decay)

  train_unet_data_psnr[i]   = to_numpy(PSNR(y_train_, image_train_)).mean()
  train_energy_data_psnr[i] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

# ======================================================================
# Testing results
# ======================================================================
(image_test, _) = next(iter(dataloader_test))
image_noise_test = degradation_func(image_test)
image_test       = image_test.to(device)
image_noise_test = image_noise_test.to(device)
y_test           = unet(image_noise_test)
y_update_test    = energy.update_prediction_langevin(image_noise_test, 
                                                     y_test.detach(), 
                                                     number_step_langevin * langevin_steps_factor, 
                                                     list_lr_langevin[-1], 
                                                     regular_data, 
                                                     add_noise, 
                                                     noise_decay)

image_test       = to_numpy(permute(image_test)).squeeze()
image_noise_test = to_numpy(permute(image_noise_test)).squeeze()
y_test           = to_numpy(permute(y_test)).squeeze()
y_update_test    = to_numpy(permute(y_update_test)).squeeze()

# -------------------------------------------------------------------
# Testing performance measuring
# -------------------------------------------------------------------
test_unet_data_psnr   = np.zeros(len(dataloader_test))
test_energy_data_psnr = np.zeros(len(dataloader_test))

for i, (image_test_, _) in enumerate(iter(dataloader_test)):
  image_noise_test_ = degradation_func(image_test_)
  image_test_       = image_test_.to(device)
  image_noise_test_ = image_noise_test_.to(device)
  y_test_           = unet(image_noise_test_)
  y_update_test_    = energy.update_prediction_langevin(image_noise_test_, 
                                                        y_test_.detach(), 
                                                        number_step_langevin * langevin_steps_factor, 
                                                        list_lr_langevin[-1], 
                                                        regular_data, 
                                                        add_noise, 
                                                        noise_decay)

  test_unet_data_psnr[i]   = to_numpy(PSNR(y_test_, image_test_)).mean()
  test_energy_data_psnr[i] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

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
unet_file_model   = os.path.join(date_model, f'unet.{time_stamp}.pth')
energy_file_model = os.path.join(date_model, f'energy.{time_stamp}.pth')

if not os.path.exists(dir_figure):
  os.makedirs(dir_figure)

if not os.path.exists(dir_option):
  os.makedirs(dir_option)

if not os.path.exists(dir_result):
  os.makedirs(dir_result)

if not os.path.exists(dir_model):
  os.makedirs(dir_model)

if not os.path.exists(path_figure):
  os.makedirs(path_figure)

if not os.path.exists(path_option):
  os.makedirs(path_option)

if not os.path.exists(path_result):
  os.makedirs(path_result)

if not os.path.exists(path_model):
  os.makedirs(path_model)

if not os.path.exists(date_figure):
  os.makedirs(date_figure)

if not os.path.exists(date_option):
  os.makedirs(date_option)

if not os.path.exists(date_result):
  os.makedirs(date_result)

if not os.path.exists(date_model):
  os.makedirs(date_model)

# -------------------------------------------------------------------
# Save models
# -------------------------------------------------------------------
torch.save(unet, unet_file_model)
torch.save(energy, energy_file_model)

# -------------------------------------------------------------------
# Save the options
# -------------------------------------------------------------------
with open(file_option, 'w') as f:
  f.write('{}: {}\n'.format('work directory', dir_work))
  f.write('{}: {}\n'.format('cuda device', cuda_device))
  f.write('{}: {}\n'.format('seed', seed))
  f.write('{}: {}\n'.format('unet out activation:', out_activation))
  f.write('{}: {}\n'.format('dataset', dir_dataset))
  f.write('{}: {}\n'.format('image size', im_size))
  f.write('{}: {}\n'.format('scale range', scale_range))
  f.write('{}: {}\n'.format('train size', train_size))
  f.write('{}: {}\n'.format('test size', test_size))
  f.write('{}: {}\n'.format('use subset', use_subset))
  f.write('{}: {}\n'.format('use label', use_label))
  f.write('{}: {}\n'.format('batch size', batch_size))
  f.write('{}: {}\n'.format('number epoch', number_epoch))
  f.write('{}: {}\n'.format('lr unet', lr_unet))
  f.write('{}: {}\n'.format('use unet scheduler', use_unet_sched))
  f.write('{}: {}\n'.format('use energy scheduler', use_energy_sched))
  f.write('{}: {}\n'.format('scheduler step size', sched_step_size))
  f.write('{}: {}\n'.format('scheduler gamma', sched_gamma))
  f.write('{}: {}\n'.format('lr energy model', lr_energy_model))
  f.write('{}: {}\n'.format('lr langevin max', lr_langevin_max))
  f.write('{}: {}\n'.format('lr langevin min', lr_langevin_min))
  f.write('{}: {}\n'.format('number step langevin', number_step_langevin))
  f.write('{}: {}\n'.format('langevin steps factor', langevin_steps_factor))
  f.write('{}: {}\n'.format('regular data', regular_data))
  f.write('{}: {}\n'.format('langevin noise decay factor', noise_decay))
  f.write('{}: {}\n'.format('use energy weights regularization', use_reg))
  f.write('{}: {}\n'.format('energy weights regularization weight', reg_weight))
  f.write('{}: {}\n'.format('use energy L2 weights regularization', use_L2_reg))
  f.write('{}: {}\n'.format('energy L2 weights regularization weight', L2_reg_weight))
  f.write('{}: {}\n'.format('add noise', add_noise))
  f.write('{}: {}\n'.format('use unpaired', use_unpaired))
  f.write('{}: {}\n'.format('sigma noise', sigma_noise))
  f.write('{}: {}\n'.format('salt and pepper noise', snp_noise))
  f.write('{}: {}\n'.format('delete square pixels', square_pixels))
  f.write('{}: {}\n'.format('degradation', degradation))
  f.write('{}: {}\n'.format('in channel', in_channel))
  f.write('{}: {}\n'.format('dim feature', dim_feature))
  f.write('{}: {}\n'.format('dim output', dim_output))
  f.write('{}: {}\n'.format('train unet psnr mean', np.mean(train_unet_data_psnr)))
  f.write('{}: {}\n'.format('train unet psnr std', np.std(train_unet_data_psnr)))
  f.write('{}: {}\n'.format('train energy psnr mean', np.mean(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('train energy psnr std', np.std(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('test unet psnr mean', np.mean(test_unet_data_psnr)))
  f.write('{}: {}\n'.format('test unet psnr std', np.std(test_unet_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr mean', np.mean(test_energy_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr std', np.std(test_energy_data_psnr)))

f.close()

# -------------------------------------------------------------------
# Save the results
# -------------------------------------------------------------------   
with open(file_result, 'w', newline='') as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(val_loss_unet_mean)
  writer.writerow(val_loss_unet_std)
  writer.writerow(val_loss_energy_model_mean)
  writer.writerow(val_loss_energy_model_std)
  writer.writerow(val_psnr_mean)
  writer.writerow(val_psnr_std)
  writer.writerow(val_psnr_langevin_mean)
  writer.writerow(val_psnr_langevin_std)
  writer.writerow(train_unet_data_psnr)
  writer.writerow(train_energy_data_psnr)
  writer.writerow(test_unet_data_psnr)
  writer.writerow(test_energy_data_psnr)

f.close()

# -------------------------------------------------------------------
# Save the figures from training
# -------------------------------------------------------------------
nRow  = 9
nCol  = 5
fSize = 3

fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('UNet')
ax[0][0].plot(val_loss_unet_mean, color='red', label='Loss', linewidth=3)
ax[0][0].legend()

ax[0][1].set_title('Energy Model')
ax[0][1].plot(val_loss_energy_model_mean, color='red', label='Loss')
ax[0][1].legend()

ax[0][2].set_title('Train PSNR')
ax[0][2].plot(val_psnr_mean, color='red', label='UNet')
ax[0][2].plot(val_psnr_langevin_mean, color='green', label='Langevin')
ax[0][2].legend()

bplot_colors = ['pink', 'lightgreen']

ax[0][3].set_title('UNet Data PSNR')
ax[0][3].yaxis.grid(True)
bplot0 = ax[0][3].boxplot([train_unet_data_psnr, 
                           test_unet_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot0['boxes'], bplot_colors):
  patch.set_facecolor(color)

ax[0][4].set_title('Energy Data PSNR')
ax[0][4].yaxis.grid(True)
bplot1 = ax[0][4].boxplot([train_energy_data_psnr, 
                           test_energy_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot1['boxes'], bplot_colors):
  patch.set_facecolor(color)

for i in range(nCol):
  ax[1][i].set_title('Train (Input)')
  if in_channel == 1:
    ax[1][i].imshow(vis_im_transform(image_noise_train[i, ...]), cmap='gray')
  else:
    ax[1][i].imshow(vis_im_transform(image_noise_train[i, ...]))

for i in range(nCol):
  ax[2][i].set_title('Train (Dataset)')
  if in_channel == 1:
    ax[2][i].imshow(vis_im_transform(image_train[i, ...]), cmap='gray')
  else:
    ax[2][i].imshow(vis_im_transform(image_train[i, ...]))

for i in range(nCol):
  ax[3][i].set_title('Train (UNet)')
  if in_channel == 1:
    ax[3][i].imshow(vis_im_transform(y_train[i, ...]), cmap='gray')
  else:
    ax[3][i].imshow(vis_im_transform(y_train[i, ...]))

for i in range(nCol):
  ax[4][i].set_title('Train (Langevin)')
  if in_channel == 1:
    ax[4][i].imshow(vis_im_transform(y_update_train[i, ...]), cmap='gray')
  else:
    ax[4][i].imshow(vis_im_transform(y_update_train[i, ...]))

for i in range(nCol):
  ax[5][i].set_title('Test (Input)')
  if in_channel == 1:
    ax[5][i].imshow(vis_im_transform(image_noise_test[i, ...]), cmap='gray')
  else:
    ax[5][i].imshow(vis_im_transform(image_noise_test[i, ...]))

for i in range(nCol):
  ax[6][i].set_title('Test (Dataset)')
  if in_channel == 1:
    ax[6][i].imshow(vis_im_transform(image_test[i, ...]), cmap='gray')
  else:
    ax[6][i].imshow(vis_im_transform(image_test[i, ...]))

for i in range(nCol):
  ax[7][i].set_title('Test (UNet)')
  if in_channel == 1:
    ax[7][i].imshow(vis_im_transform(y_test[i, ...]), cmap='gray')
  else:
    ax[7][i].imshow(vis_im_transform(y_test[i, ...]))

for i in range(nCol):
  ax[8][i].set_title('Test (Langevin)')
  if in_channel == 1:
    ax[8][i].imshow(vis_im_transform(y_update_test[i, ...]), cmap='gray')
  else:
    ax[8][i].imshow(vis_im_transform(y_update_test[i, ...]))

plt.tight_layout()
fig.savefig(file_figure, bbox_inches='tight', dpi=600)
plt.close(fig)