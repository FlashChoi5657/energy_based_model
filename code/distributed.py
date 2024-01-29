import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime, random
import csv, os, argparse

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from models import CondEnergyModel, EnergyModel, UNet
from ResNet import IGEBM
from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256
from sn_gan import Discriminator
from utils import permute, to_numpy, init_weights, UNet_Energy_log, get_dataloaders, SGLD
from utils import gaussian_noise, sp_noise, delete_square, generate_Y0, clip_grad, SampleBuffer

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='20m_4m')
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--in_channel", type=int, default=3)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--dim_feature", type=int, default=64)

parser.add_argument("--gaussian_noise", type=float, default=0.25)
parser.add_argument("--sp_noise", type=float, default=0.1)
parser.add_argument("--square_pixels", type=int, default=10)

parser.add_argument("--lr_energy_model", type=float, default=0.00005)
parser.add_argument("--lr_langevin_min", type=float, default=0.000006)
parser.add_argument("--lr_langevin_max", type=float, default=0.00001)
parser.add_argument("--number_step_langevin", type=int, default=6)

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_plot", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

parser.add_argument("--model_param_load", type=bool, default=False)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

        
def main():
  ngpu = torch.cuda.device_count()    
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  scale_range = [-1, 1] # [0,1]

  # ======================================================================
  # Instantiations
  # ======================================================================
  # energy = IGEBM(args.in_channel, args.dim_feature, args.dim_output, activation='silu', scale_range=scale_range,
                      # use_L2_reg=args.use_en_L2_reg, L2_reg_weight=args.en_L2_reg_w)
  # energy = Discriminator(args.in_channel, args.dim_feature)
  energy = EBM_CelebA64(args.in_channel, args.dim_feature)
  # energy = EnergyModel(in_channel, dim_feature, dim_output, activation='silu', 
  #                      use_gp=True, scale_range=scale_range,
  #                      use_L2_reg=args.use_en_L2_reg, L2_reg_weight=L2_reg_weight )
  unet = UNet(args.in_channel, args.dim_feature // 2, args.in_channel)

  # ======================================================================
  # Dataloaders 
  # ======================================================================

  dir_data = '/hdd1/dataset'
  num_worker = 4 * ngpu
  transform = torchvision.transforms.Compose([ 
      torchvision.transforms.Resize(args.img_size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  print('==> Preparing data..')
  trainset = torchvision.datasets.CIFAR10(dir_data, train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(dir_data, train=False, download=True, transform=transform)
  dl_train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_worker)
  dl_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_worker)

  if ngpu > 1:    
    print('==> Making model..')
    energy = nn.DataParallel(energy)
    energy = energy.to(device)
    unet = nn.DataParallel(unet)
    unet = unet.to(device)

    num_params = sum(p.numel() for p in energy.parameters() if p.requires_grad)
    print('The number of parameters of energy model is', num_params)

    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print('The number of parameters of unet is', num_params)

  else:
    energy = energy.cuda()
    unet = unet.cuda()

  # use buffer
  # buffer = SampleBuffer()

  # ======================================================================
  # Weights initializations
  # ======================================================================
  # TODO: implement weights initialization method

  if args.model_param_load:
    model_path = '/nas/users/minhyeok/energy_based_model/experiments/model/CIFAR10/2023_05_08/energy.17_26_27.pth'
    energy = torch.load(model_path)
  else:
    # energy.apply(init_weights)
    unet.apply(init_weights)
    pass
    
  # ======================================================================
  # Optimizers
  # ======================================================================
  optim_energy = torch.optim.Adam(energy.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
  # optim_energy = torch.optim.AdamW(energy.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
  optim_unet = torch.optim.AdamW(unet.parameters(), lr=0.00001, betas=(args.b1, 0.999))

  criterion = nn.MSELoss(reduce='sum')
  # ======================================================================
  # Schedulers
  # ======================================================================
  # ExponentialLR
  # sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=sched_step_size, gamma=0.93)
  # sched_optim_energy = torch.optim.lr_scheduler.ExponentialLR(optim_energy, gamma=0.93)
  sched_optim_energy = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda = lambda epoch: 0.95 ** epoch)
  
  # sched_optim_unet = torch.optim.lr_scheduler.ExponentialLR(optim_unet, gamma=0.93)
  sched_optim_unet = torch.optim.lr_scheduler.LambdaLR(optim_unet, lr_lambda = lambda epoch: 0.95 ** epoch)
  
  
  train(energy, unet, args, optim_energy, sched_optim_energy, optim_unet, criterion, sched_optim_unet, dl_train, dl_test, device, ngpu)
  
  
  
  
  
  

def train(energy, unet, args, optim_energy, sched_optim_energy, optim_unet, criterion, sched_optim_unet, dataloader_train, dataloader_test, device, ngpu):
  # ======================================================================
  # Options
  # ======================================================================
  # curr_dir_path         = os.path.join(os.path.dirname(os.path.abspath('')), '..', 'experiments')
  dir_work              = '/nas/users/minhyeok/energy_based_model/experiments'
  name                  = args.name
  cuda_device           = 0
  batch_size            = args.batch_size
  number_epoch          = args.epochs
  use_energy_sched      = True
  sigma_noise           = args.gaussian_noise
  sched_step_size       = number_epoch // 10
  lr_langevin_max       = args.lr_langevin_max
  lr_langevin_min       = args.lr_langevin_min
  number_step_langevin  = args.number_step_langevin
  in_channel            = args.in_channel
  dim_feature           = args.dim_feature
  seed                  = args.seed
  list_lr_langevin      = np.linspace(lr_langevin_max, lr_langevin_min, num=number_epoch, endpoint=True)
  pl.seed_everything(seed)

  # ======================================================================
  # Dataset 
  # ======================================================================
  # dir_data    = os.path.join(dir_work, 'data')
  dir_dataset = args.dataset

  # ======================================================================
  # Path for the results
  # ======================================================================
  dir_figure = os.path.join(dir_work, 'figure')
  dir_result = os.path.join(dir_work, 'result')
  dir_model  = os.path.join(dir_work, 'model')
  now        = datetime.datetime.now()
  date_stamp = now.strftime('%Y_%m_%d')
  time_stamp = now.strftime('%H_%M_%S')

  path_figure = os.path.join(dir_figure, dir_dataset)
  path_result = os.path.join(dir_result, dir_dataset)
  path_model  = os.path.join(dir_model, dir_dataset)

  date_figure = os.path.join(path_figure, date_stamp)
  date_result = os.path.join(path_result, date_stamp)
  date_model  = os.path.join(path_model, date_stamp)

  # file_figure       = os.path.join(date_figure, f'{time_stamp}.png')
  file_result       = os.path.join(date_result, f'{time_stamp}.csv')
  energy_file_model = os.path.join(date_model, f'energy.{time_stamp}.pth')
  unet_file_model = os.path.join(date_model, f'unet.{time_stamp}.pth')
    
  if not os.path.exists(dir_figure):  os.makedirs(dir_figure)

  if not os.path.exists(dir_result):  os.makedirs(dir_result)

  if not os.path.exists(dir_model):  os.makedirs(dir_model)

  if not os.path.exists(path_figure):  os.makedirs(path_figure)

  if not os.path.exists(path_result):  os.makedirs(path_result)

  if not os.path.exists(path_model):  os.makedirs(path_model)

  if not os.path.exists(date_figure):  os.makedirs(date_figure)

  if not os.path.exists(date_result):  os.makedirs(date_result)

  if not os.path.exists(date_model):  os.makedirs(date_model)
    
  # ======================================================================
  # Evaluation
  # ======================================================================
  PSNR = PeakSignalNoiseRatio().to(device)

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
  # torch.autograd.set_detect_anomaly(True)
  cnt=0
  for i in range(number_epoch):
    val_loss_energy_model = list()
    val_ori               = list()
    val_psnr              = list()
    val_psnr_langevin     = list()
    val_loss_unet         = list()

    for j, (image, _) in enumerate(iter(dataloader_train)):
      # -------------------------------------------------------------------
      # Applied degradation
      # -------------------------------------------------------------------

      image_noise = gaussian_noise(image, sigma_noise)
      # image_noise = image + sigma_noise * torch.randn_like(image)
      # if len(buffer) < 1:
      #   buffer.push(image_noise)
      image       = image.to(device)
      image_noise = image_noise.to(device)
        
      # -------------------------------------------------------------------
      # Y0 generation identity, gaussian distribution, zero
      # -------------------------------------------------------------------
      unet.train()
      optim_unet.zero_grad()
      lr_unet_model = sched_optim_unet.get_last_lr()[0]
      
      prediction = unet(image_noise)
      # loss_u = unet.compute_loss(prediction, image)
      loss_u = criterion(prediction, image)
      
      loss_u.backward()
      optim_unet.step()
      
      sched_optim_unet.step()
      # prediction = generate_Y0(image_noise, 'id')
      unet.eval()

      # -------------------------------------------------------------------
      # Yn generation
      # -------------------------------------------------------------------
      energy.eval()
      for p in energy.parameters():
        p.requires_grad = False
      
      # update = image_noise.clone()
      # update.requires_grad = True
      # update = update.to(device)
      prediction_update = SGLD(prediction.detach(), energy, number_step_langevin, list_lr_langevin[i])

      # prediction_update = prediction.clone()
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
    lr_energy_model = sched_optim_energy.get_last_lr()[0]

    if use_energy_sched:
      sched_optim_energy.step()

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
                                                    lr_unet_model, 
                                                    lr_energy_model, 
                                                    val_psnr_ori_mean[i],
                                                    val_psnr_mean[i], 
                                                    val_psnr_langevin_mean[i])
    print(f"{name}:\n", log)
    
    col = 8
    if val_psnr_langevin_mean[i] > 27:
      cnt +=1
    if i % args.save_plot == 0 or cnt == 1:
      fig, ax = plt.subplots(4, col, figsize = (3 * 15, 3 * col))
      for k in range(col):
        if ngpu > 1:
          rand = k
        else:
          rand = random.randrange(args.batch_size)
        ax[0][k].set_title('Train (Input)')
        ax[0][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
        ax[1][k].set_title(f'Train (Noisy) :{100*sigma_noise}')
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

    if val_psnr_langevin_mean[i] < val_psnr_langevin_mean[i- int(number_epoch/10)] or cnt == 1:
    # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
      print(f'Terminating the run after {i} epochs..')
      break
    
    if val_psnr_langevin_mean[i] < 10:
      print("learning break.. ")
      break
      
      
  # ======================================================================
  # Model evaluation for inferences
  # ======================================================================
  energy = energy.eval()
  unet = unet.eval()

  # ======================================================================
  # Training results
  # ======================================================================
  (image_train, _) = next(iter(dataloader_train))
  image_noise_train = gaussian_noise(image_train, sigma_noise)
  # image_train       = image_train.to(device)
  image_noise_train = image_noise_train.to(device)
  y_train           = unet(image_noise_train)
  # y_train           = image_noise_train.clone()
  y_update_train    = SGLD(y_train.detach(), energy, number_step_langevin, list_lr_langevin[-1])


  # -------------------------------------------------------------------
  # Training performance measuring
  # -------------------------------------------------------------------
  train_self_data_psnr     = np.zeros(len(dataloader_train))
  train_energy_data_psnr = np.zeros(len(dataloader_train))

  for j, (image_train_, _) in enumerate(iter(dataloader_train)):
    image_noise_train_ = gaussian_noise(image_train_, sigma_noise)
    image_train_       = image_train_.to(device)
    image_noise_train_ = image_noise_train_.to(device)
    y_train_           = unet(image_noise_train_)
    # y_train_           = image_noise_train_.clone()
    y_update_train_    = SGLD(y_train_.detach(), energy, number_step_langevin, list_lr_langevin[-1])

    train_self_data_psnr[j]     = to_numpy(PSNR(y_train_.detach(), image_train_)).mean()
    train_energy_data_psnr[j] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

  # ======================================================================
  # Testing results
  # ======================================================================
  (image_test, _) = next(iter(dataloader_test))
  image_noise_test = gaussian_noise(image_test, sigma_noise)
  image_test       = image_test.to(device)
  image_noise_test = image_noise_test.to(device)
  y_test           = unet(image_noise_test)
  # y_test           = image_noise_test.clone()
  y_update_test    = SGLD(y_test.detach(), energy, number_step_langevin, list_lr_langevin[-1])

  # ---------------------------------------------------clip_grad-------------
  test_self_data_psnr     = np.zeros(len(dataloader_test))
  test_energy_data_psnr = np.zeros(len(dataloader_test))

  for j, (image_test_, _) in enumerate(iter(dataloader_test)):
    image_noise_test_ = gaussian_noise(image_test_, sigma_noise)
    # image_test_       = image_test_.to(device)
    image_noise_test_ = image_noise_test_.to(device)
    y_test_           = unet(image_noise_test_)
    # y_test_           = image_noise_test_.clone()
    y_update_test_    = SGLD(y_test_.detach(), energy, number_step_langevin, list_lr_langevin[-1])

    test_self_data_psnr[j]     = to_numpy(PSNR(y_test_.detach(), image_test_)).mean()
    test_energy_data_psnr[j] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

  # -------------------------------------------------------------------
  # Save models
  # -------------------------------------------------------------------
  torch.save(energy, energy_file_model)
  torch.save(unet, unet_file_model)

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

  ax[0][3].set_title('ID Data PSNR')
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
    if ngpu > 1:
      rand = k
    else:
      rand = random.randrange(args.batch_size)
    ax[1][k].set_title('Train (Input)')
    ax[1][k].imshow(torchvision.utils.make_grid(image_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
    ax[2][k].set_title(f'Train (Noisy): {100*sigma_noise}')
    ax[2][k].imshow(torchvision.utils.make_grid(image_noise_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
    ax[3][k].set_title('Train (UNet)')
    ax[3][k].imshow(torchvision.utils.make_grid(y_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
    ax[4][k].set_title('Train (Langevin)')
    ax[4][k].imshow(torchvision.utils.make_grid(y_update_train[rand].detach().cpu(), normalize=True).permute(1,2,0))

    ax[5][k].set_title('Test (Input)')
    ax[5][k].imshow(torchvision.utils.make_grid(image_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
    ax[6][k].set_title(f'Test (Noisy): {100*sigma_noise}')
    ax[6][k].imshow(torchvision.utils.make_grid(image_noise_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
    ax[7][k].set_title('Test (UNet)')
    ax[7][k].imshow(torchvision.utils.make_grid(y_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
    ax[8][k].set_title('Test (Langevin)')
    ax[8][k].imshow(torchvision.utils.make_grid(y_update_test[rand].detach().cpu(), normalize=True).permute(1,2,0))

  time_1 = datetime.datetime.now().strftime('%HH_%MM')
  plt.tight_layout()
  fig.savefig(f'{date_figure}/last_{name}_{time_1}.png', bbox_inches='tight', dpi=600)
  plt.close(fig)
  
if __name__ == '__main__':
  main()