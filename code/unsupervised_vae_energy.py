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
from models import VAE, CondEnergyModel
from energy_based_model.experiments.utils import permute, to_numpy, init_weights, VAE_Energy_log, get_unsup_dataloaders
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
parser.add_argument("--dim_feature", type=int, default=32)
parser.add_argument("--dim_latent", type=int, default=128)
parser.add_argument("--gaussian_noise", type=float, default=0.75)
parser.add_argument("--sp_noise", type=float, default=0.1)
parser.add_argument("--square_pixels", type=int, default=20)
parser.add_argument("--degradation", type=str, default='gaussian_noise')
parser.add_argument("--lr_vae", type=float, default=0.005)
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
parser.add_argument("--train_ratio", type=float, default=0.1)
parser.add_argument("--use_unpaired", action="store_true", default=False)
parser.add_argument("--use_unsup_unpaired", action="store_true", default=False)
parser.add_argument("--use_energy_as_loss", action="store_true", default=False)
parser.add_argument("--use_unsup_subset", action="store_true", default=False)
parser.add_argument("--use_unsup_label", type=int, default=5)
parser.add_argument("--regular_data", type=float, default=0.0)
parser.add_argument("--use_energy_reg", action="store_true", default=False)
parser.add_argument("--energy_reg_weight", type=float, default=0.0)
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
lr_vae                = args.lr_vae
use_vae_sched         = False
use_energy_sched      = args.use_energy_sched
sched_step_size       = 10
sched_gamma           = 0.93
lr_energy_model       = args.lr_energy_model
lr_langevin_max       = args.lr_langevin_max
lr_langevin_min       = args.lr_langevin_min
number_step_langevin  = args.number_step_langevin
langevin_steps_factor = args.langevin_steps_factor
regular_data          = args.regular_data
use_reg               = args.use_energy_reg
reg_weight            = args.energy_reg_weight
add_noise             = True
use_energy_as_loss    = args.use_energy_as_loss
use_unpaired          = args.use_unpaired
use_unsup_unpaired    = args.use_unsup_unpaired # works only with use_energy_as_loss=False
in_channel            = args.in_channel
dim_latent            = args.dim_latent
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
dir_data         = os.path.join(dir_work, 'data')
dir_dataset      = args.dataset
im_size          = 32
scale_range      = [-1, 1]
train_size       = args.train_size
test_size        = args.test_size
use_subset       = args.use_subset
use_label        = args.use_label
use_unsup_subset = args.use_unsup_subset
use_unsup_label  = args.use_unsup_label
train_ratio      = args.train_ratio

# ======================================================================
# Dataloaders 
# ======================================================================
dataloader_train, dataloader_unsup_train, dataloader_test, out_activation, vis_im_transform = get_unsup_dataloaders(dir_data, dir_dataset, 
                                                                                                                    im_size, in_channel, 
                                                                                                                    batch_size, train_size, 
                                                                                                                    train_ratio, test_size, 
                                                                                                                    use_subset, use_label, 
                                                                                                                    scale_range, use_unsup_subset, 
                                                                                                                    use_unpaired)

# ======================================================================
# Evaluation
# ======================================================================
PSNR = PeakSignalNoiseRatio().to(device)

# ======================================================================
# Instantiations
# ======================================================================
vae    = VAE(in_channel, dim_feature, dim_latent, vae_out).to(device)
energy = CondEnergyModel(in_channel * 2, dim_feature, dim_output, scale_range=scale_range, 
                                                                  use_reg=use_reg, 
                                                                  reg_weight=reg_weight).to(device)

# ======================================================================
# Weights initializations
# ======================================================================
vae.apply(init_weights)
# TODO: implement weights initialization method
# energy.apply(init_weights)

# ======================================================================
# Optimizers
# ======================================================================
optim_vae    = torch.optim.Adam(vae.parameters(), lr=lr_vae)
optim_energy = torch.optim.Adam(energy.parameters(), lr=lr_energy_model)

# ======================================================================
# Schedulers
# ======================================================================
# ExponentialLR
sched_optim_vae    = torch.optim.lr_scheduler.StepLR(optim_vae, step_size=sched_step_size, gamma=sched_gamma)
sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=sched_step_size, gamma=sched_gamma)

# ======================================================================
# Variables for the results 
# ======================================================================

val_loss_vae_mean                = np.zeros(number_epoch)
val_loss_vae_std                 = np.zeros(number_epoch)
val_loss_vae_data_fidelity_mean  = np.zeros(number_epoch)
val_loss_vae_data_fidelity_std   = np.zeros(number_epoch)
val_loss_vae_regularization_mean = np.zeros(number_epoch)
val_loss_vae_regularization_std  = np.zeros(number_epoch)
val_loss_energy_model_mean       = np.zeros(number_epoch)
val_loss_energy_model_std        = np.zeros(number_epoch)
val_psnr_mean                    = np.zeros(number_epoch)
val_psnr_std                     = np.zeros(number_epoch)
val_psnr_langevin_mean           = np.zeros(number_epoch)
val_psnr_langevin_std            = np.zeros(number_epoch)

# ======================================================================
# Training
# ======================================================================
vae    = vae.train()
energy = energy.train()

for i in range(number_epoch):
  val_loss_vae                = list()
  val_loss_vae_data_fidelity  = list()
  val_loss_vae_regularization = list()
  val_loss_energy_model       = list()
  val_psnr                    = list()
  val_psnr_langevin           = list()
  
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
      image_noise = degradation_func(image)

    image       = image.to(device)
    image_noise = image_noise.to(device)

    # -------------------------------------------------------------------
    # Y0 generation
    # -------------------------------------------------------------------
    prediction, mu, log_var, z = vae(image_noise)

    # -------------------------------------------------------------------
    # Yn generation
    # -------------------------------------------------------------------
    prediction_update = energy.update_prediction_langevin(image_noise, 
                                                          prediction.detach(), 
                                                          number_step_langevin,  
                                                          list_lr_langevin[i], 
                                                          regular_data, 
                                                          add_noise)
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
    # VAE gradient evaluation
    # -------------------------------------------------------------------
    optim_vae.zero_grad()
    loss_vae, loss_vae_recon, loss_vae_kld = vae.compute_loss(prediction, mu, log_var, prediction_update.detach())
    loss_vae.backward()
    
    value_psnr          = to_numpy(PSNR(prediction, image))
    value_psnr_langevin = to_numpy(PSNR(prediction_update, image))

    # -------------------------------------------------------------------
    # Update networks
    # -------------------------------------------------------------------
    optim_vae.step()
    optim_energy.step()
    
    # -------------------------------------------------------------------
    # Save results for each batch iteration
    # -------------------------------------------------------------------        
    val_loss_vae.append(loss_vae.item())
    val_loss_vae_data_fidelity.append(loss_vae_recon.item())
    val_loss_vae_regularization.append(loss_vae_kld.item())
    val_loss_energy_model.append(loss_energy.item())
    val_psnr.append(value_psnr)
    val_psnr_langevin.append(value_psnr_langevin)

  # -------------------------------------------------------------------
  # Update schedulers
  # -------------------------------------------------------------------
  lr_vae          = sched_optim_vae.get_last_lr()[0]
  lr_energy_model = sched_optim_energy.get_last_lr()[0]

  if use_vae_sched:
    sched_optim_vae.step()
  if use_energy_sched:
    sched_optim_energy.step()

  # -------------------------------------------------------------------
  # Save results for each epoch
  # -------------------------------------------------------------------        
  val_loss_vae_mean[i]                = np.mean(val_loss_vae)
  val_loss_vae_std[i]                 = np.std(val_loss_vae)
  val_loss_vae_data_fidelity_mean[i]  = np.mean(val_loss_vae_data_fidelity)
  val_loss_vae_data_fidelity_std[i]   = np.std(val_loss_vae_data_fidelity)
  val_loss_vae_regularization_mean[i] = np.mean(val_loss_vae_regularization)
  val_loss_vae_regularization_std[i]  = np.std(val_loss_vae_regularization)
  val_loss_energy_model_mean[i]       = np.mean(val_loss_energy_model)
  val_loss_energy_model_std[i]        = np.std(val_loss_energy_model)
  val_psnr_mean[i]                    = np.mean(val_psnr)
  val_psnr_std[i]                     = np.std(val_psnr)
  val_psnr_langevin_mean[i]           = np.mean(val_psnr_langevin)
  val_psnr_langevin_std[i]            = np.std(val_psnr_langevin)

  log = VAE_Energy_log(number_step_langevin) % (i, 
                                                number_epoch, 
                                                val_loss_vae_mean[i], 
                                                lr_vae, 
                                                val_loss_vae_data_fidelity_mean[i], 
                                                val_loss_vae_regularization_mean[i], 
                                                val_loss_energy_model_mean[i], 
                                                lr_energy_model, 
                                                val_psnr_mean[i], 
                                                val_psnr_langevin_mean[i])
  print(f"{name}:\n", log)
  
  # TODO: find a condition to break the loop
  # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
  #   print(f'Terminating the run after {i} epochs..')
  #   break

# ======================================================================
# Model evaluation for inferences
# ======================================================================
vae    = vae.eval()
energy = energy.eval()

# ======================================================================
# Training results
# ======================================================================
(image_train, _) = next(iter(dataloader_train))
image_noise_train = degradation_func(image_train)
image_train       = image_train.to(device)
image_noise_train = image_noise_train.to(device)

y_train, mu_train, log_var_train, z_train = vae.forward(image_noise_train)
y_update_train                            = energy.update_prediction_langevin(image_noise_train, 
                                                                              y_train.detach(), 
                                                                              number_step_langevin, 
                                                                              list_lr_langevin[-1], 
                                                                              regular_data, 
                                                                              add_noise)

z_train           = to_numpy(z_train).squeeze()
image_train       = to_numpy(permute(image_train)).squeeze()
image_noise_train = to_numpy(permute(image_noise_train)).squeeze()
y_train           = to_numpy(permute(y_train)).squeeze()
y_update_train    = to_numpy(permute(y_update_train)).squeeze()

# -------------------------------------------------------------------
# Training performance measuring
# -------------------------------------------------------------------
train_vae_data_psnr    = np.zeros(len(dataloader_train))
train_energy_data_psnr = np.zeros(len(dataloader_train))

for i, (image_train_, _) in enumerate(iter(dataloader_train)):
  image_noise_train_ = degradation_func(image_train_)
  image_train_       = image_train_.to(device)
  image_noise_train_ = image_noise_train_.to(device)
  
  y_train_, mu_train_, log_var_train_, z_train_ = vae.forward(image_noise_train_)
  y_update_train_                               = energy.update_prediction_langevin(image_noise_train_, 
                                                                                    y_train_.detach(), 
                                                                                    number_step_langevin, 
                                                                                    list_lr_langevin[-1], 
                                                                                    regular_data, 
                                                                                    add_noise)
  train_vae_data_psnr[i]    = to_numpy(PSNR(y_train_, image_train_)).mean()
  train_energy_data_psnr[i] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

# ======================================================================
# Testing results
# ======================================================================
(image_test, _) = next(iter(dataloader_test))
image_noise_test = degradation_func(image_test)
image_test       = image_test.to(device)
image_noise_test = image_noise_test.to(device)

y_test, mu_test, log_var_test, z_test = vae.forward(image_noise_test)
y_update_test                         = energy.update_prediction_langevin(image_noise_test, 
                                                                          y_test.detach(), 
                                                                          number_step_langevin, 
                                                                          list_lr_langevin[-1], 
                                                                          regular_data, 
                                                                          add_noise)

z_test           = to_numpy(z_test).squeeze()
image_test       = to_numpy(permute(image_test)).squeeze()
image_noise_test = to_numpy(permute(image_noise_test)).squeeze()
y_test           = to_numpy(permute(y_test)).squeeze()
y_update_test    = to_numpy(permute(y_update_test)).squeeze()

# -------------------------------------------------------------------
# Testing performance measuring
# -------------------------------------------------------------------
test_vae_data_psnr    = np.zeros(len(dataloader_test))
test_energy_data_psnr = np.zeros(len(dataloader_test))

for i, (image_test_, _) in enumerate(iter(dataloader_test)):
  image_noise_test_ = degradation_func(image_test_)
  image_test_       = image_test_.to(device)
  image_noise_test_ = image_noise_test_.to(device)

  y_test_, mu_test_, log_var_test_, z_test_ = vae.forward(image_noise_test_)
  y_update_test_                            = energy.update_prediction_langevin(image_noise_test_, 
                                                                                y_test_.detach(), 
                                                                                number_step_langevin, 
                                                                                list_lr_langevin[-1], 
                                                                                regular_data, 
                                                                                add_noise)
  test_vae_data_psnr[i]    = to_numpy(PSNR(y_test_, image_test_)).mean()
  test_energy_data_psnr[i] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

# ======================================================================
# Results from latent
# ======================================================================
y = to_numpy(vae.sample(batch_size, device)).squeeze()

# ======================================================================
# VAE UNSUPERVISED TRAINING - using a fixed energy model weights
# ======================================================================
vae = vae.train()

# ======================================================================
# Optimizers for unsupervised training
# ======================================================================
lr_vae    = args.lr_vae
optim_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae)

# ======================================================================
# Schedulers for unsupervised training
# ======================================================================
# ExponentialLR
sched_optim_vae = torch.optim.lr_scheduler.StepLR(optim_vae, step_size=sched_step_size, gamma=sched_gamma)

if use_unsup_unpaired:
  energy             = energy.train()
  lr_energy_model    = args.lr_energy_model
  optim_energy       = torch.optim.Adam(filter(lambda p: p.requires_grad, energy.parameters()), lr=lr_energy_model)
  sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=sched_step_size, gamma=sched_gamma)

# ======================================================================
# Variables for the unsupervised results 
# ======================================================================
val_unsup_loss_vae_mean                = np.zeros(number_epoch)
val_unsup_loss_vae_std                 = np.zeros(number_epoch)
val_unsup_loss_vae_data_fidelity_mean  = np.zeros(number_epoch)
val_unsup_loss_vae_data_fidelity_std   = np.zeros(number_epoch)
val_unsup_loss_vae_regularization_mean = np.zeros(number_epoch)
val_unsup_loss_vae_regularization_std  = np.zeros(number_epoch)
val_unsup_loss_energy_model_mean       = np.zeros(number_epoch)
val_unsup_loss_energy_model_std        = np.zeros(number_epoch)
val_unsup_psnr_mean                    = np.zeros(number_epoch)
val_unsup_psnr_std                     = np.zeros(number_epoch)
val_unsup_psnr_langevin_mean           = np.zeros(number_epoch)
val_unsup_psnr_langevin_std            = np.zeros(number_epoch)

for i in range(number_epoch):
  val_loss_vae                = list()
  val_loss_vae_data_fidelity  = list()
  val_loss_vae_regularization = list()
  val_loss_energy_model       = list()
  val_psnr                    = list()
  val_psnr_langevin           = list()
  
  for j, (image, _) in enumerate(iter(dataloader_unsup_train)):
    # -------------------------------------------------------------------
    # Gaussian noise
    # -------------------------------------------------------------------
    if use_unpaired:
      image, image2 = torch.split(image, batch_size)
      image2_noise  = degradation_func(image2)
      image2        = image2.to(device)
      image2_noise  = image2_noise.to(device)

    else:
      image_noise = degradation_func(image)

    image       = image.to(device)
    image_noise = image_noise.to(device)

    # -------------------------------------------------------------------
    # Y0 generation
    # -------------------------------------------------------------------
    prediction, mu, log_var, z = vae(image_noise)

    # -------------------------------------------------------------------
    # Yn generation
    # -------------------------------------------------------------------
    prediction_update = energy.update_prediction_langevin(image_noise, 
                                                          prediction.detach(), 
                                                          number_step_langevin, 
                                                          list_lr_langevin[i], 
                                                          regular_data, 
                                                          add_noise)

    # -------------------------------------------------------------------
    # VAE gradient evaluation
    # -------------------------------------------------------------------
    optim_vae.zero_grad()
    loss_vae, loss_vae_recon, loss_vae_kld = vae.compute_loss(prediction, mu, log_var, prediction_update.detach())

    if use_energy_as_loss:
      optim_energy.zero_grad()
      loss_energy = energy.compute_loss_inference(image_noise, prediction)
      loss_energy.backward()

    else:
      if use_unsup_unpaired:
        (image_unpaired, _)  = next(iter(dataloader_train))
        image_noise_unpaired = degradation_func(image_unpaired)
        image_unpaired       = image_unpaired.to(device)
        image_noise_unpaired = image_noise_unpaired.to(device)

        prediction_unpaired, mu_unpaired, log_var_unpaired, z_unpaired = vae(image_noise_unpaired)
        prediction_update_unpaired                                     = energy.update_prediction_langevin(image_noise_unpaired, 
                                                                                                           prediction_unpaired.detach(), 
                                                                                                           number_step_langevin, 
                                                                                                           list_lr_langevin[i], 
                                                                                                           regular_data, 
                                                                                                           add_noise)

        optim_energy.zero_grad()
        loss_energy = energy.compute_loss(image_noise_unpaired, image_unpaired, image_noise_unpaired, prediction_update_unpaired.detach())
        loss_energy.backward()

      else:
        loss_energy = energy.compute_loss_inference(image_noise, prediction_update.detach())
      loss_vae.backward()

    value_psnr          = to_numpy(PSNR(prediction, image))
    value_psnr_langevin = to_numpy(PSNR(prediction_update, image))

    # -------------------------------------------------------------------
    # Update networks
    # -------------------------------------------------------------------
    optim_vae.step()
    if use_unsup_unpaired:
      optim_energy.step()
    
    # -------------------------------------------------------------------
    # Save unsupervised results for each batch iteration
    # -------------------------------------------------------------------        
    val_loss_vae.append(loss_vae.item())
    val_loss_vae_data_fidelity.append(loss_vae_recon.item())
    val_loss_vae_regularization.append(loss_vae_kld.item())
    val_loss_energy_model.append(loss_energy.item())
    val_psnr.append(value_psnr)
    val_psnr_langevin.append(value_psnr_langevin)

  # -------------------------------------------------------------------
  # Update scheduler
  # -------------------------------------------------------------------
  lr_vae = sched_optim_vae.get_last_lr()[0]
  if use_vae_sched:
    sched_optim_vae.step()

  lr_energy_model = sched_optim_energy.get_last_lr()[0]
  if use_energy_sched and use_unsup_unpaired:
    sched_optim_energy.step()

  # -------------------------------------------------------------------
  # Save unsupervised results for each epoch
  # -------------------------------------------------------------------        
  val_unsup_loss_vae_mean[i]                = np.mean(val_loss_vae)
  val_unsup_loss_vae_std[i]                 = np.std(val_loss_vae)
  val_unsup_loss_vae_data_fidelity_mean[i]  = np.mean(val_loss_vae_data_fidelity)
  val_unsup_loss_vae_data_fidelity_std[i]   = np.std(val_loss_vae_data_fidelity)
  val_unsup_loss_vae_regularization_mean[i] = np.mean(val_loss_vae_regularization)
  val_unsup_loss_vae_regularization_std[i]  = np.std(val_loss_vae_regularization)
  val_unsup_loss_energy_model_mean[i]       = np.mean(val_loss_energy_model)
  val_unsup_loss_energy_model_std[i]        = np.std(val_loss_energy_model)
  val_unsup_psnr_mean[i]                    = np.mean(val_psnr)
  val_unsup_psnr_std[i]                     = np.std(val_psnr)
  val_unsup_psnr_langevin_mean[i]           = np.mean(val_psnr_langevin)
  val_unsup_psnr_langevin_std[i]            = np.std(val_psnr_langevin)

  log = VAE_Energy_log(number_step_langevin) % (i, 
                                                number_epoch, 
                                                val_unsup_loss_vae_mean[i], 
                                                lr_vae, 
                                                val_unsup_loss_vae_data_fidelity_mean[i], 
                                                val_unsup_loss_vae_regularization_mean[i], 
                                                val_unsup_loss_energy_model_mean[i], 
                                                lr_energy_model, 
                                                val_unsup_psnr_mean[i], 
                                                val_unsup_psnr_langevin_mean[i])
  print(f"{name}:\n", log)
  
  # TODO: find a condition to break the loop
  # if np.isnan(val_unsup_psnr_mean[i]) or ((i > 10) and (val_unsup_psnr_mean[i] < 5)) or ((i > 100) and (val_unsup_psnr_mean[i] < 10)):
  #   print(f'Terminating the unsupervised training run after {i} epochs..')
  #   break

# ======================================================================
# Model evaluation for unsupervised inferences
# ======================================================================
vae    = vae.eval()
energy = energy.eval()

# ======================================================================
# Unsupervised Training results
# ======================================================================
(image_unsup_train, _)  = next(iter(dataloader_train))
image_noise_unsup_train = degradation_func(image_unsup_train)
image_unsup_train       = image_unsup_train.to(device)
image_noise_unsup_train = image_noise_unsup_train.to(device)

y_unsup_train, mu_unsup_train, log_var_unsup_train, z_unsup_train = vae.forward(image_noise_unsup_train)
y_update_unsup_train                                              = energy.update_prediction_langevin(image_noise_unsup_train, 
                                                                                                      y_unsup_train.detach(), 
                                                                                                      number_step_langevin * langevin_steps_factor, 
                                                                                                      list_lr_langevin[-1], 
                                                                                                      regular_data, 
                                                                                                      add_noise)

z_unsup_train           = to_numpy(z_unsup_train).squeeze()
image_unsup_train       = to_numpy(image_unsup_train).squeeze()
image_noise_unsup_train = to_numpy(image_noise_unsup_train).squeeze()
y_unsup_train           = to_numpy(y_unsup_train).squeeze()
y_update_unsup_train    = to_numpy(y_update_unsup_train).squeeze()

# -------------------------------------------------------------------
# Unsupervised Training performance measuring
# -------------------------------------------------------------------
unsup_train_vae_data_psnr    = np.zeros(len(dataloader_train))
unsup_train_energy_data_psnr = np.zeros(len(dataloader_train))

for i, (image_unsup_train_, _) in enumerate(iter(dataloader_train)):
  image_noise_unsup_train_ = degradation_func(image_unsup_train_)
  image_unsup_train_       = image_unsup_train_.to(device)
  image_noise_unsup_train_ = image_noise_unsup_train_.to(device)
  
  y_unsup_train_, mu_unsup_train_, log_var_unsup_train_, z_unsup_train_ = vae.forward(image_noise_unsup_train_)
  y_update_unsup_train_                                                 = energy.update_prediction_langevin(image_noise_unsup_train_, 
                                                                                                            y_unsup_train_.detach(), 
                                                                                                            number_step_langevin * langevin_steps_factor, 
                                                                                                            list_lr_langevin[-1], 
                                                                                                            regular_data, 
                                                                                                            add_noise)
  unsup_train_vae_data_psnr[i]    = to_numpy(PSNR(y_unsup_train_, image_unsup_train_)).mean()
  unsup_train_energy_data_psnr[i] = to_numpy(PSNR(y_update_unsup_train_, image_unsup_train_)).mean()

# ======================================================================
# Testing results
# ======================================================================
(image_unsup_test, _)  = next(iter(dataloader_test))
image_noise_unsup_test = degradation_func(image_unsup_test)
image_unsup_test       = image_unsup_test.to(device)
image_noise_unsup_test = image_noise_unsup_test.to(device)

y_unsup_test, mu_unsup_test, log_var_unsup_test, z_unsup_test = vae.forward(image_noise_unsup_test)
y_update_unsup_test                                           = energy.update_prediction_langevin(image_noise_unsup_test, 
                                                                                                  y_unsup_test.detach(), 
                                                                                                  number_step_langevin * langevin_steps_factor, 
                                                                                                  list_lr_langevin[-1], 
                                                                                                  regular_data, 
                                                                                                  add_noise)

z_unsup_test           = to_numpy(z_unsup_test).squeeze()
image_unsup_test       = to_numpy(permute(image_unsup_test)).squeeze()
image_noise_unsup_test = to_numpy(image_noise_unsup_test).squeeze()
y_unsup_test           = to_numpy(permute(y_unsup_test)).squeeze()
y_update_unsup_test    = to_numpy(permute(y_update_unsup_test)).squeeze()

# -------------------------------------------------------------------
# Testing performance measuring
# -------------------------------------------------------------------
unsup_test_vae_data_psnr    = np.zeros(len(dataloader_test))
unsup_test_energy_data_psnr = np.zeros(len(dataloader_test))

for i, (image_unsup_test_, _) in enumerate(iter(dataloader_test)):
  image_noise_unsup_test_ = degradation_func(image_unsup_test_)
  image_unsup_test_       = image_unsup_test_.to(device)
  image_noise_unsup_test_ = image_noise_unsup_test_.to(device)

  y_unsup_test_, mu_unsup_test_, log_var_unsup_test_, z_unsup_test_ = vae.forward(image_noise_unsup_test_)
  y_update_unsup_test_                                              = energy.update_prediction_langevin(image_noise_unsup_test_, 
                                                                                                        y_unsup_test_.detach(), 
                                                                                                        number_step_langevin * langevin_steps_factor, 
                                                                                                        list_lr_langevin[-1], 
                                                                                                        regular_data, 
                                                                                                        add_noise)
  unsup_test_vae_data_psnr[i]    = to_numpy(PSNR(y_unsup_test_, image_unsup_test_)).mean()
  unsup_test_energy_data_psnr[i] = to_numpy(PSNR(y_update_unsup_test_, image_unsup_test_)).mean()

# ======================================================================
# Results from unsupervised latent
# ======================================================================
unsup_y = to_numpy(vae.sample(batch_size, device)).squeeze()

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
vae_file_model    = os.path.join(date_model, f'vae.{time_stamp}.pth')
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
torch.save(vae, vae_file_model)
torch.save(energy, energy_file_model)

# -------------------------------------------------------------------
# Save the options
# -------------------------------------------------------------------         
with open(file_option, 'w') as f:
  f.write('{}: {}\n'.format('work directory', dir_work))
  f.write('{}: {}\n'.format('cuda device', cuda_device))
  f.write('{}: {}\n'.format('seed', seed))
  f.write('{}: {}\n'.format('vae out activation', vae_out))
  f.write('{}: {}\n'.format('dataset', dir_dataset))
  f.write('{}: {}\n'.format('image size', im_size))
  f.write('{}: {}\n'.format('scale range', scale_range))
  f.write('{}: {}\n'.format('train size', train_size))
  f.write('{}: {}\n'.format('test size', test_size))
  f.write('{}: {}\n'.format('use subset', use_subset))
  f.write('{}: {}\n'.format('use label', use_label))
  f.write('{}: {}\n'.format('batch size', batch_size))
  f.write('{}: {}\n'.format('number epoch', number_epoch))
  f.write('{}: {}\n'.format('lr vae', lr_vae))
  f.write('{}: {}\n'.format('use vae scheduler', use_vae_sched))
  f.write('{}: {}\n'.format('use energy scheduler', use_energy_sched))
  f.write('{}: {}\n'.format('scheduler step size', sched_step_size))
  f.write('{}: {}\n'.format('scheduler gamma', sched_gamma))
  f.write('{}: {}\n'.format('lr energy model', lr_energy_model))
  f.write('{}: {}\n'.format('lr langevin max', lr_langevin_max))
  f.write('{}: {}\n'.format('lr langevin min', lr_langevin_min))
  f.write('{}: {}\n'.format('number step langevin', number_step_langevin))
  f.write('{}: {}\n'.format('langevin steps factor', langevin_steps_factor))
  f.write('{}: {}\n'.format('regular data', regular_data))
  f.write('{}: {}\n'.format('add noise', add_noise))
  f.write('{}: {}\n'.format('use energy value as VAE loss', use_energy_as_loss))
  f.write('{}: {}\n'.format('use unpaired for unsupervised training', use_unsup_unpaired))
  f.write('{}: {}\n'.format('use unpaired', use_unpaired))
  f.write('{}: {}\n'.format('sigma noise', sigma_noise))
  f.write('{}: {}\n'.format('salt and pepper noise', snp_noise))
  f.write('{}: {}\n'.format('delete square pixels', square_pixels))
  f.write('{}: {}\n'.format('degradation', degradation))
  f.write('{}: {}\n'.format('in channel', in_channel))
  f.write('{}: {}\n'.format('dim latent', dim_latent))
  f.write('{}: {}\n'.format('dim feature', dim_feature))
  f.write('{}: {}\n'.format('dim output', dim_output))
  f.write('{}: {}\n'.format('train vae psnr mean', np.mean(train_vae_data_psnr)))
  f.write('{}: {}\n'.format('train vae psnr std', np.std(train_vae_data_psnr)))
  f.write('{}: {}\n'.format('train energy psnr mean', np.mean(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr std', np.std(train_energy_data_psnr)))
  f.write('{}: {}\n'.format('test vae psnr mean', np.mean(test_vae_data_psnr)))
  f.write('{}: {}\n'.format('test vae psnr std', np.std(test_vae_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr mean', np.mean(test_energy_data_psnr)))
  f.write('{}: {}\n'.format('test energy psnr std', np.std(test_energy_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised train vae psnr mean', np.mean(unsup_train_vae_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised train vae psnr std', np.std(unsup_train_vae_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised train energy psnr mean', np.mean(unsup_train_energy_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised train energy psnr std', np.std(unsup_train_energy_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised test vae psnr mean', np.mean(unsup_test_vae_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised test vae psnr std', np.std(unsup_test_vae_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised test energy psnr mean', np.mean(unsup_test_energy_data_psnr)))
  f.write('{}: {}\n'.format('unsupervised test energy psnr std', np.std(unsup_test_energy_data_psnr)))

f.close()

# -------------------------------------------------------------------
# Save the results
# -------------------------------------------------------------------   
with open(file_result, 'w', newline='') as f:
  writer  = csv.writer(f, delimiter=',')
  writer.writerow(val_loss_vae_mean)
  writer.writerow(val_loss_vae_std)
  writer.writerow(val_loss_vae_data_fidelity_mean)
  writer.writerow(val_loss_vae_data_fidelity_std)
  writer.writerow(val_loss_vae_regularization_mean)
  writer.writerow(val_loss_vae_regularization_std)
  writer.writerow(val_loss_energy_model_mean)
  writer.writerow(val_loss_energy_model_std)
  writer.writerow(val_psnr_mean)
  writer.writerow(val_psnr_std)
  writer.writerow(val_psnr_langevin_mean)
  writer.writerow(val_psnr_langevin_std)
  writer.writerow(val_unsup_loss_vae_mean)
  writer.writerow(val_unsup_loss_vae_std)
  writer.writerow(val_unsup_loss_vae_data_fidelity_mean)
  writer.writerow(val_unsup_loss_vae_data_fidelity_std)
  writer.writerow(val_unsup_loss_vae_regularization_mean)
  writer.writerow(val_unsup_loss_vae_regularization_std)
  writer.writerow(val_unsup_loss_energy_model_mean)
  writer.writerow(val_unsup_loss_energy_model_std)
  writer.writerow(val_unsup_psnr_mean)
  writer.writerow(val_unsup_psnr_std)
  writer.writerow(val_unsup_psnr_langevin_mean)
  writer.writerow(val_unsup_psnr_langevin_std)
  writer.writerow(train_vae_data_psnr)
  writer.writerow(train_energy_data_psnr)
  writer.writerow(test_vae_data_psnr)
  writer.writerow(test_energy_data_psnr)
  writer.writerow(unsup_train_vae_data_psnr)
  writer.writerow(unsup_train_energy_data_psnr)
  writer.writerow(unsup_test_vae_data_psnr)
  writer.writerow(unsup_test_energy_data_psnr)

f.close()

# -------------------------------------------------------------------
# Save the figures from trainings
# -------------------------------------------------------------------
nRow  = 12
nCol  = 8 
fSize = 3

fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

ax[0][0].set_title('Supervised VAE')
ax[0][0].plot(val_loss_vae_mean, color='red', label='Loss', linewidth=3)
ax[0][0].plot(val_loss_vae_data_fidelity_mean, color='green', label='Data Fidelity')
ax[0][0].plot(val_loss_vae_regularization_mean, color='blue', label='Regularization')
ax[0][0].legend()

ax[0][1].set_title('Unsupervised VAE')
ax[0][1].plot(val_unsup_loss_vae_mean, color='red', label='Loss', linewidth=3)
ax[0][1].plot(val_unsup_loss_vae_data_fidelity_mean, color='green', label='Data Fidelity')
ax[0][1].plot(val_unsup_loss_vae_regularization_mean, color='blue', label='Regularization')
ax[0][1].legend()

ax[0][2].set_title('Supervised Energy Model')
ax[0][2].plot(val_loss_energy_model_mean, color='red', label='Loss')
ax[0][2].legend()

ax[0][3].set_title('Unsupervised Energy Model')
ax[0][3].plot(val_unsup_loss_energy_model_mean, color='red', label='Loss')
ax[0][3].legend()

ax[0][4].set_title('Train Accuracy (PSNR)')
ax[0][4].plot(val_psnr_mean, color='red', label='VAE')
ax[0][4].plot(val_psnr_langevin_mean, color='blue', label='Langevin')
ax[0][4].legend()

ax[0][5].set_title('Unsupervised Train Accuracy (PSNR)')
ax[0][5].plot(val_unsup_psnr_mean, color='red', label='VAE')
ax[0][5].plot(val_unsup_psnr_langevin_mean, color='green', label='Langevin')
ax[0][5].legend()

bplot_colors = ['pink', 'lightgreen']

ax[0][6].set_title('Test Data VAEs PSNR')
ax[0][6].yaxis.grid(True)
bplot0 = ax[0][6].boxplot([test_vae_data_psnr, 
                           unsup_test_vae_data_psnr], 0, vert=True, patch_artist=True, labels=['VAE(0)', 'VAE(1)'])
for patch, color in zip(bplot0['boxes'], bplot_colors):
  patch.set_facecolor(color)

ax[0][7].set_title('Test Data Energys PSNR')
ax[0][7].yaxis.grid(True)
bplot1 = ax[0][7].boxplot([test_energy_data_psnr, 
                           unsup_test_energy_data_psnr], 0, vert=True, patch_artist=True, labels=['Energy(0)', 'Energy(1)'])
for patch, color in zip(bplot1['boxes'], bplot_colors):
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
  ax[3][i].set_title('Training (VAE)')
  if in_channel == 1:
    ax[3][i].imshow(vis_im_transform(y_train[i, ...]), cmap='gray')
  else:
    ax[3][i].imshow(vis_im_transform(y_train[i, ...]))

for i in range(nCol):
  ax[4][i].set_title('Training (Langevin)')
  if in_channel == 1:
    ax[4][i].imshow(vis_im_transform(y_update_train[i, ...]), cmap='gray')
  else:
    ax[4][i].imshow(vis_im_transform(y_update_train[i, ...]))

for i in range(nCol):
  ax[5][i].set_title('Testing (Input)')
  if in_channel == 1:
    ax[5][i].imshow(vis_im_transform(image_noise_test[i, ...]), cmap='gray')
  else:
    ax[5][i].imshow(vis_im_transform(image_noise_test[i, ...]))

for i in range(nCol):
  ax[6][i].set_title('Testing (Dataset)')
  if in_channel == 1:
    ax[6][i].imshow(vis_im_transform(image_test[i, ...]), cmap='gray')
  else:
    ax[6][i].imshow(vis_im_transform(image_test[i, ...]))

for i in range(nCol):
  ax[7][i].set_title('Testing (VAE)')
  if in_channel == 1:
    ax[7][i].imshow(vis_im_transform(y_test[i, ...]), cmap='gray')
  else:
    ax[7][i].imshow(vis_im_transform(y_test[i, ...]))

for i in range(nCol):
  ax[8][i].set_title('Testing (Langevin)')
  if in_channel == 1:
    ax[8][i].imshow(vis_im_transform(y_update_test[i, ...]), cmap='gray')
  else:
    ax[8][i].imshow(vis_im_transform(y_update_test[i, ...]))

for i in range(nCol):
  ax[9][i].set_title('Unsupervised Testing (Dataset)')
  if in_channel == 1:
    ax[9][i].imshow(vis_im_transform(image_unsup_test[i, ...]), cmap='gray')
  else:
    ax[9][i].imshow(vis_im_transform(image_unsup_test[i, ...]))

for i in range(nCol):
  ax[10][i].set_title('Unsupervised Testing (VAE)')
  if in_channel == 1:
    ax[10][i].imshow(vis_im_transform(y_unsup_test[i, ...]), cmap='gray')
  else:
    ax[10][i].imshow(vis_im_transform(y_unsup_test[i, ...]))

for i in range(nCol):
  ax[11][i].set_title('Unsupervised Testing (Langevin)')
  if in_channel == 1:
    ax[11][i].imshow(vis_im_transform(y_update_unsup_test[i, ...]), cmap='gray')
  else:
    ax[11][i].imshow(vis_im_transform(y_update_unsup_test[i, ...]))

plt.tight_layout()
fig.savefig(file_figure, bbox_inches='tight', dpi=600)
plt.close(fig)