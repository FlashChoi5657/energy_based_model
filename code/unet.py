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
from models import UNet
from energy_based_model.experiments.utils import permute, to_numpy, init_weights, UNet_log, get_dataloaders
from energy_based_model.experiments.utils import gaussian_noise, sp_noise, delete_square

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--train_size", type=int, default=60000)
parser.add_argument("--test_size", type=int, default=10000)
parser.add_argument("--in_channel", type=int, default=1)
parser.add_argument("--gaussian_noise", type=float, default=0.75)
parser.add_argument("--sp_noise", type=float, default=0.1)
parser.add_argument("--square_pixels", type=int, default=20)
parser.add_argument("--degradation", type=str, default='gaussian_noise')
parser.add_argument("--lr_unet", type=float, default=0.005)
parser.add_argument("--use_subset", action="store_true", default=False)
parser.add_argument("--use_label", type=int, default=4)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--train_ratio", type=float, default=1.0)
args = parser.parse_args()

# ======================================================================
# Options
# ======================================================================
curr_dir_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
name            = args.name
dir_work        = os.path.join(curr_dir_path, name)
cuda_device     = 0
device          = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
batch_size      = args.batch_size
number_epoch    = args.epochs
lr_unet         = args.lr_unet
use_sched       = False
sched_step_size = 10
sched_gamma     = 0.93
in_channel      = args.in_channel
degradation     = args.degradation
sigma_noise     = args.gaussian_noise
snp_noise       = args.sp_noise
square_pixels   = args.square_pixels
seed            = 0
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
                                                                                      use_subset, use_label, scale_range)

# ======================================================================
# Evaluation
# ======================================================================
PSNR = PeakSignalNoiseRatio().to(device)

# ======================================================================
# Instantiations
# ======================================================================
unet = UNet(in_channel=in_channel, out_channel=in_channel, bilinear=False, out_activation=out_activation).to(device)

# ======================================================================
# Weights initialization
# ======================================================================
# unet.apply(init_weights)

# ======================================================================
# Optimizer
# ======================================================================
optim_unet = torch.optim.Adam(unet.parameters(), lr=lr_unet)

# ======================================================================
# Scheduler
# ======================================================================
# ExponentialLR
sched_optim_unet = torch.optim.lr_scheduler.StepLR(optim_unet, step_size=sched_step_size, gamma=sched_gamma)

# ======================================================================
# Variables for the results 
# ======================================================================
val_loss_unet_mean = np.zeros(number_epoch)
val_loss_unet_std  = np.zeros(number_epoch)
val_psnr_mean      = np.zeros(number_epoch)
val_psnr_std       = np.zeros(number_epoch)

# ======================================================================
# Training
# ======================================================================
unet = unet.train()

for i in range(number_epoch):
  val_loss_unet = list()
  val_psnr      = list()

  for j, (image, _) in enumerate(iter(dataloader_train)):
    # -------------------------------------------------------------------
    # Applied degradation
    # -------------------------------------------------------------------
    image_noise = degradation_func(image)
    image       = image.to(device)
    image_noise = image_noise.to(device)

    # -------------------------------------------------------------------
    # Y0 generation
    # -------------------------------------------------------------------
    prediction = unet(image_noise)

    # -------------------------------------------------------------------
    # Gradient Evaluation
    # -------------------------------------------------------------------
    optim_unet.zero_grad()
    loss_unet = unet.compute_loss(prediction, image)
    loss_unet.backward() 
    
    value_psnr = to_numpy(PSNR(prediction, image)).mean()

    # -------------------------------------------------------------------
    # Update network
    # -------------------------------------------------------------------
    optim_unet.step()
    
    # -------------------------------------------------------------------
    # Save results for each batch iteration
    # -------------------------------------------------------------------
    val_loss_unet.append(loss_unet.item()) 
    val_psnr.append(value_psnr)

  # -------------------------------------------------------------------
  # Update scheduler
  # -------------------------------------------------------------------
  lr_unet = sched_optim_unet.get_last_lr()[0]
  if use_sched:
    sched_optim_unet.step()

  # -------------------------------------------------------------------
  # Save results for each epoch
  # -------------------------------------------------------------------        
  val_loss_unet_mean[i] = np.mean(val_loss_unet)
  val_loss_unet_std[i]  = np.std(val_loss_unet)
  val_psnr_mean[i]      = np.mean(val_psnr)
  val_psnr_std[i]       = np.std(val_psnr)
  
  log = UNet_log() % (i, 
                      number_epoch, 
                      val_loss_unet_mean[i], 
                      lr_unet, 
                      val_psnr_mean[i])
  print(f"{name}:\n", log)
  
  # TODO: find a condition to break the loop
  # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)) or ((i > 100) and (val_psnr_mean[i] < 13)):
  #   print(f'Terminating the run after {i} epochs..')
  #   break

# ======================================================================
# Model evaluation for inferences
# ======================================================================
unet = unet.eval()

# ======================================================================
# Training results
# ======================================================================
(image_train, _) = next(iter(dataloader_train))
image_noise_train = degradation_func(image_train)
image_train       = image_train.to(device)
image_noise_train = image_noise_train.to(device)
y_train           = unet(image_noise_train)

image_train       = to_numpy(permute(image_train)).squeeze()
image_noise_train = to_numpy(permute(image_noise_train)).squeeze()
y_train           = to_numpy(permute(y_train)).squeeze()

# -------------------------------------------------------------------
# Training performance measuring
# -------------------------------------------------------------------
train_data_psnr = np.zeros(len(dataloader_train))

for i, (image_train_, _) in enumerate(iter(dataloader_train)):
  image_noise_train_ = degradation_func(image_train_)
  image_train_       = image_train_.to(device)
  image_noise_train_ = image_noise_train_.to(device)
  y_train_           = unet(image_noise_train_)

  train_data_psnr[i] = to_numpy(PSNR(y_train_, image_train_)).mean()

# ======================================================================
# Testing results
# ======================================================================
(image_test, _) = next(iter(dataloader_test))
image_noise_test = degradation_func(image_test)
image_test       = image_test.to(device)
image_noise_test = image_noise_test.to(device)
y_test           = unet(image_noise_test)

image_test       = to_numpy(permute(image_test)).squeeze()
image_noise_test = to_numpy(permute(image_noise_test)).squeeze()
y_test           = to_numpy(permute(y_test)).squeeze()

# -------------------------------------------------------------------
# Testing performance measuring
# -------------------------------------------------------------------
test_data_psnr = np.zeros(len(dataloader_test))

for i, (image_test_, _) in enumerate(iter(dataloader_test)):
  image_noise_test_ = degradation_func(image_test_)  
  image_test_       = image_test_.to(device)
  image_noise_test_ = image_noise_test_.to(device)
  y_test_           = unet(image_noise_test_)
  
  test_data_psnr[i] = to_numpy(PSNR(y_test_, image_test_)).mean()

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

file_figure     = os.path.join(date_figure, '{}.png'.format(time_stamp))
file_option     = os.path.join(date_option, '{}.ini'.format(time_stamp))
file_result     = os.path.join(date_result, '{}.csv'.format(time_stamp))
unet_file_model = os.path.join(date_model, 'unet.{}.pth'.format(time_stamp))

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
# Save the model
# -------------------------------------------------------------------
torch.save(unet, unet_file_model)

# -------------------------------------------------------------------
# Save the options
# -------------------------------------------------------------------         
with open(file_option, 'w') as f:
  f.write('{}: {}\n'.format('work directory', dir_work))
  f.write('{}: {}\n'.format('cuda device', cuda_device))
  f.write('{}: {}\n'.format('seed', seed))
  f.write('{}: {}\n'.format('dataset', dir_dataset))
  f.write('{}: {}\n'.format('image size', im_size))
  f.write('{}: {}\n'.format('unet out activation:', out_activation))
  f.write('{}: {}\n'.format('scale range', scale_range))
  f.write('{}: {}\n'.format('train ratio', train_ratio))
  f.write('{}: {}\n'.format('train size', train_size))
  f.write('{}: {}\n'.format('test size', test_size))
  f.write('{}: {}\n'.format('use subset', use_subset))
  f.write('{}: {}\n'.format('use label', use_label))
  f.write('{}: {}\n'.format('batch size', batch_size))
  f.write('{}: {}\n'.format('number epoch', number_epoch))
  f.write('{}: {}\n'.format('lr unet', lr_unet))
  f.write('{}: {}\n'.format('use scheduler', use_sched))
  f.write('{}: {}\n'.format('scheduler step size', sched_step_size))
  f.write('{}: {}\n'.format('scheduler gamma', sched_gamma))
  f.write('{}: {}\n'.format('sigma noise', sigma_noise))
  f.write('{}: {}\n'.format('salt and pepper noise', snp_noise))
  f.write('{}: {}\n'.format('delete square pixels', square_pixels))
  f.write('{}: {}\n'.format('degradation', degradation))
  f.write('{}: {}\n'.format('in channel', in_channel))
  f.write('{}: {}\n'.format('train psnr mean', np.mean(train_data_psnr)))
  f.write('{}: {}\n'.format('train psnr mean', np.std(train_data_psnr)))
  f.write('{}: {}\n'.format('test psnr mean', np.mean(test_data_psnr)))
  f.write('{}: {}\n'.format('test psnr mean', np.std(test_data_psnr)))

f.close()

# -------------------------------------------------------------------
# Save the results
# -------------------------------------------------------------------         
with open(file_result, 'w', newline='') as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(val_loss_unet_mean)
  writer.writerow(val_loss_unet_std)
  writer.writerow(val_psnr_mean)
  writer.writerow(val_psnr_std)
  writer.writerow(train_data_psnr)
  writer.writerow(test_data_psnr)

f.close()

# -------------------------------------------------------------------
# Save the figures from training
# -------------------------------------------------------------------
nRow  = 7 
nCol  = 4 
fSize = 3
  
fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('UNet')
ax[0][0].plot(val_loss_unet_mean, color='red', label='Loss', linewidth=3)
ax[0][0].legend()

ax[0][1].set_title('Train Accuracy (PSNR)')
ax[0][1].plot(val_psnr_mean, color='red', label='UNet')
ax[0][1].legend()

bplot_colors = ['pink', 'lightgreen']

ax[0][2].set_title('Data UNet PSNR')
ax[0][2].yaxis.grid(True)
bplot0 = ax[0][2].boxplot([train_data_psnr, 
                           test_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
for patch, color in zip(bplot0['boxes'], bplot_colors):
  patch.set_facecolor(color)

for i in range(nCol):
  ax[1][i].set_title('Training (Input)')
  if in_channel == 1:
    ax[1][i].imshow(vis_im_transform(image_noise_train[i, ...]), cmap='gray')
  else:
    ax[1][i].imshow(vis_im_transform(image_noise_train[i, ...]))

for i in range(nCol):
  ax[2][i].set_title('Training (Dataset)')
  if in_channel == 1:
    ax[2][i].imshow(vis_im_transform(image_train[i, ...]), cmap='gray')
  else:
    ax[2][i].imshow(vis_im_transform(image_train[i, ...]))

for i in range(nCol):
  ax[3][i].set_title('Training (UNet)')
  if in_channel == 1:
    ax[3][i].imshow(vis_im_transform(y_train[i, ...]), cmap='gray')
  else:
    ax[3][i].imshow(vis_im_transform(y_train[i, ...]))

for i in range(nCol):
  ax[4][i].set_title('Testing (Input)')
  if in_channel == 1:
    ax[4][i].imshow(vis_im_transform(image_noise_test[i, ...]), cmap='gray')
  else:
    ax[4][i].imshow(vis_im_transform(image_noise_test[i, ...]))

for i in range(nCol):
  ax[5][i].set_title('Testing (Dataset)')
  if in_channel == 1:
    ax[5][i].imshow(vis_im_transform(image_test[i, ...]), cmap='gray')
  else:
    ax[5][i].imshow(vis_im_transform(image_test[i, ...]))

for i in range(nCol):
  ax[6][i].set_title('Testing (UNet)')
  if in_channel == 1:
    ax[6][i].imshow(vis_im_transform(y_test[i, ...]), cmap='gray')
  else:
    ax[6][i].imshow(vis_im_transform(y_test[i, ...]))

plt.tight_layout()
fig.savefig(file_figure, bbox_inches='tight', dpi=600)
plt.close(fig)