import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime, random
import csv, os, argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader


from models import CondEnergyModel, EnergyModel, UNet
from ResNet import IGEBM
from sn_gan import Discriminator
from utils import permute, to_numpy, init_weights, UNet_Energy_log, get_dataloaders, SGLD
from utils import gaussian_noise, sp_noise, delete_square, generate_Y0, clip_grad, SampleBuffer

# ======================================================================
# Input arguments
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='param_17m_u_30m')
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_size", type=int, default=60000)
parser.add_argument("--test_size", type=int, default=10000)

parser.add_argument("--in_channel", type=int, default=3)
# parser.add_argument("--dim_output", type=int, default=1)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--dim_feature", type=int, default=128)

parser.add_argument("--gaussian_noise", type=float, default=0.25)
# parser.add_argument("--sp_noise", type=float, default=0.1)
# parser.add_argument("--square_pixels", type=int, default=10)
# parser.add_argument("--degradation", type=str, default='gaussian_noise')
# parser.add_argument("--Y0_type", type=str, default='id')

parser.add_argument("--lr_energy_model", type=float, default=0.0001)
parser.add_argument("--lr_langevin_min", type=float, default=0.00001)
parser.add_argument("--lr_langevin_max", type=float, default=0.00008)
parser.add_argument("--number_step_langevin", type=int, default=6)
parser.add_argument("--sched_gamma", type=float, default=0.93)

# parser.add_argument("--use_subset", action="store_true", default=False)
# parser.add_argument("--use_label", type=int, default=5)

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=40)

parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

# parser.add_argument("--train_ratio", type=float, default=1.0)
# parser.add_argument("--use_unpaired", action="store_true", default=False)
# parser.add_argument("--regular_data", type=float, default=0.0)
parser.add_argument("--init_noise_decay", type=float, default=1.0)
# parser.add_argument("--use_gp", action="store_true", default=True)
# parser.add_argument("--use_energy_reg", action="store_true", default=False)
# parser.add_argument("--energy_reg_weight", type=float, default=0.0005)
# parser.add_argument("--use_en_L2_reg", action="store_true", default=False)
# parser.add_argument("--en_L2_reg_w", type=float, default=0.001)

parser.add_argument("--model_param_load", type=bool, default=False)

parser.add_argument('--num_workers', type=int, default=2, help='')

args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

def ddp_setup(rank, world_size):
  """
  Args:
      rank: Unique identifier of each process
      world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

class Trainer:
  def __init__(
    self,
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    gpu_id: int,
    save_every: int,
) -> None:
    self.gpu_id = gpu_id
    self.model = model.to(gpu_id)
    self.train_data = train_data
    self.optimizer = optimizer
    self.save_every = save_every
    self.model = DDP(model, device_ids=[gpu_id])

  def _run_batch(self, source, targets):
    self.optimizer.zero_grad()
    output = self.model(source)
    loss = F.cross_entropy(output, targets)
    loss.backward()
    self.optimizer.step()

  def _run_epoch(self, epoch):
    b_sz = len(next(iter(self.train_data))[0])
    print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
    self.train_data.sampler.set_epoch(epoch)
    for source, targets in self.train_data:
      source = source.to(self.gpu_id)
      targets = targets.to(self.gpu_id)
      self._run_batch(source, targets)

  def train(self, max_epochs: int):
    for epoch in range(max_epochs):
      self._run_epoch(epoch)
      if self.gpu_id == 0 and epoch % self.save_every == 0:
        self._save_checkpoint(epoch)
                
def load_train_objs(args):
  dir_data = '/hdd1/dataset'
  transform = torchvision.transforms.Compose([ 
      torchvision.transforms.Resize(args.img_size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
      # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    
  train_set = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=True, download=True)
  test_set = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=False, download=True)

  energy = Discriminator(args.in_channel, args.dim_feature)
  unet = UNet(args.in_channel, args.dim_feature // 4, args.in_channel)
  optim_energy = torch.optim.Adam(energy.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
  # optim_energy = torch.optim.AdamW(energy.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
  optim_unet = torch.optim.AdamW(unet.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))

  return train_set, energy, optim_energy


def prepare_dataloader(dataset: Dataset, batch_size: int):
  return DataLoader(
      dataset,
      batch_size=batch_size,
      pin_memory=True,
      shuffle=False,
      sampler=DistributedSampler(dataset)
  )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
  ddp_setup(rank, world_size)
  dataset, model, optimizer = load_train_objs()
  train_data = prepare_dataloader(dataset, batch_size)
  trainer = Trainer(model, train_data, optimizer, rank, save_every)
  trainer.train(total_epochs)
  destroy_process_group()
    
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, default='param_17m_u_30m')
  parser.add_argument("--dataset", type=str, default="CIFAR10")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--train_size", type=int, default=60000)
  parser.add_argument("--test_size", type=int, default=10000)

  parser.add_argument("--in_channel", type=int, default=3)
  # parser.add_argument("--dim_output", type=int, default=1)
  parser.add_argument("--img_size", type=int, default=64)
  parser.add_argument("--dim_feature", type=int, default=128)

  parser.add_argument("--gaussian_noise", type=float, default=0.25)
  # parser.add_argument("--sp_noise", type=float, default=0.1)
  # parser.add_argument("--square_pixels", type=int, default=10)
  # parser.add_argument("--degradation", type=str, default='gaussian_noise')
  # parser.add_argument("--Y0_type", type=str, default='id')

  parser.add_argument("--lr_energy_model", type=float, default=0.0001)
  parser.add_argument("--lr_langevin_min", type=float, default=0.00001)
  parser.add_argument("--lr_langevin_max", type=float, default=0.00008)
  parser.add_argument("--number_step_langevin", type=int, default=6)
  parser.add_argument("--sched_gamma", type=float, default=0.93)

  # parser.add_argument("--use_subset", action="store_true", default=False)
  # parser.add_argument("--use_label", type=int, default=5)

  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--save_every", type=int, default=10)
  parser.add_argument("--batch_size", type=int, default=40)

  parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

  # parser.add_argument("--train_ratio", type=float, default=1.0)
  # parser.add_argument("--use_unpaired", action="store_true", default=False)
  # parser.add_argument("--regular_data", type=float, default=0.0)
  parser.add_argument("--init_noise_decay", type=float, default=1.0)
  # parser.add_argument("--use_gp", action="store_true", default=True)
  # parser.add_argument("--use_energy_reg", action="store_true", default=False)
  # parser.add_argument("--energy_reg_weight", type=float, default=0.0005)
  # parser.add_argument("--use_en_L2_reg", action="store_true", default=False)
  # parser.add_argument("--en_L2_reg_w", type=float, default=0.001)

  parser.add_argument("--model_param_load", type=bool, default=False)

  parser.add_argument('--num_workers', type=int, default=2, help='')

  args = parser.parse_args()
  
  world_size = torch.cuda.device_count()
  mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
  
        
# def main():
#   ngpu = torch.cuda.device_count()    
#   device = 'cuda' if torch.cuda.is_available() else 'cpu'
#   scale_range = [-1, 1] # [0,1]
#   # ======================================================================
#   # Model
#   # ======================================================================
#   # energy = IGEBM(args.in_channel, args.dim_feature, args.dim_output, activation='silu', scale_range=scale_range,
#   #                     use_L2_reg=args.use_en_L2_reg, L2_reg_weight=args.en_L2_reg_w)
#   energy = Discriminator(args.in_channel, args.dim_feature)
#   # energy = EnergyModel(in_channel, dim_feature, dim_output, activation='silu', 
#   #                      use_gp=True, scale_range=scale_range,
#   #                      use_L2_reg=args.use_en_L2_reg, L2_reg_weight=L2_reg_weight )
#   unet = UNet(args.in_channel, args.dim_feature // 4, args.in_channel)
  
#   # ======================================================================
#   # Dataloaders 
#   # ======================================================================

#   dir_data = '/hdd1/dataset'
#   args.num_worker = 4 * ngpu
#   print('==> Preparing data..')
  
#   transform = torchvision.transforms.Compose([ 
#       torchvision.transforms.Resize(args.img_size),
#       torchvision.transforms.ToTensor(),
#       torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
#       # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#   ])
  
#   dataset_train = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=True, download=True)
#   dataset_test = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=False, download=True)
  
#   if ngpu > 1:    
    
#     print('==> Making model..')
#     energy = DDP(energy, device_ids=[gpu_id])
#     energy = energy.to(device)
#     unet = nn.DataParallel(unet)
#     unet = unet.to(device)
#     num_params = sum(p.numel() for p in energy.parameters() if p.requires_grad)
#     print('The number of parameters of energy model is', num_params)

#     num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
#     print('The number of parameters of unet is', num_params)

#   else:
#     energy = energy.cuda()

#   # use buffer
#   # buffer = SampleBuffer()

#   # ======================================================================
#   # Weights initializations
#   # ======================================================================
#   # TODO: implement weights initialization method

#   if args.model_param_load:
#     model_path = '/nas/users/minhyeok/energy_based_model/experiments/model/CIFAR10/2023_05_08/energy.17_26_27.pth'
#     energy = torch.load(model_path)
#   else:
#     # energy.apply(init_weights)
#     unet.apply(init_weights)
#     pass
    
#   # ======================================================================
#   # Optimizers
#   # ======================================================================
#   optim_energy = torch.optim.Adam(energy.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
#   # optim_energy = torch.optim.AdamW(energy.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
#   optim_unet = torch.optim.AdamW(unet.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))

#   criterion = nn.MSELoss(reduce='sum')
#   # ======================================================================
#   # Schedulers
#   # ======================================================================
#   # ExponentialLR
#   # sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=sched_step_size, gamma=sched_gamma)
#   # sched_optim_energy = torch.optim.lr_scheduler.ExponentialLR(optim_energy, gamma=args.sched_gamma)
#   sched_optim_energy = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda = lambda epoch: 0.95 ** epoch)
  
#   # sched_optim_unet = torch.optim.lr_scheduler.ExponentialLR(optim_unet, gamma=args.sched_gamma)
#   sched_optim_unet = torch.optim.lr_scheduler.LambdaLR(optim_unet, lr_lambda = lambda epoch: 0.95 ** epoch)
  
  
#   train(energy, unet, args, optim_energy, sched_optim_energy, optim_unet, criterion, sched_optim_unet, dl_train, dl_test, device, ngpu)
  
  
  
  
  
  

# def train(energy, unet, args, optim_energy, sched_optim_energy, optim_unet, criterion, sched_optim_unet, dataloader_train, dataloader_test, device, ngpu):
#   # ======================================================================
#   # Options
#   # ======================================================================
#   # curr_dir_path         = os.path.join(os.path.dirname(os.path.abspath('')), '..', 'experiments')
#   dir_work              = '/nas/users/minhyeok/energy_based_model/experiments'
#   name                  = args.name
#   cuda_device           = 0
#   batch_size            = args.batch_size
#   number_epoch          = args.epochs
#   use_energy_sched      = True
#   sched_step_size       = number_epoch // 10
#   sched_gamma           = args.sched_gamma
#   lr_energy_model       = args.lr_energy_model
#   lr_langevin_max       = args.lr_langevin_max
#   lr_langevin_min       = args.lr_langevin_min
#   number_step_langevin  = args.number_step_langevin
#   regular_data          = args.regular_data
#   init_noise_decay      = args.init_noise_decay
#   # use_reg               = args.use_energy_reg
#   reg_weight            = args.energy_reg_weight
#   # use_L2_reg            = args.use_energy_L2_reg
#   L2_reg_weight         = args.en_L2_reg_w
#   add_noise             = True
#   use_unpaired          = args.use_unpaired
#   in_channel            = args.in_channel
#   dim_feature           = args.dim_feature
#   dim_output            = args.dim_output
#   degradation           = args.degradation
#   Y0_type               = args.Y0_type
#   sigma_noise           = args.gaussian_noise
#   snp_noise             = args.sp_noise
#   square_pixels         = args.square_pixels
#   # use_gp                = args.use_gp
#   seed                  = args.seed
#   list_lr_langevin      = np.linspace(lr_langevin_max, lr_langevin_min, num=number_epoch, endpoint=True)
#   pl.seed_everything(seed)

#   # ======================================================================
#   # Dataset 
#   # ======================================================================
#   # dir_data    = os.path.join(dir_work, 'data')
#   dir_dataset = args.dataset
#   im_size     = args.img_size
#   scale_range = [-1, 1] # [0,1]
#   train_size  = args.train_size
#   test_size   = args.test_size
#   use_subset  = args.use_subset
#   use_label   = args.use_label

#   # ======================================================================
#   # Path for the results
#   # ======================================================================
#   dir_figure = os.path.join(dir_work, 'figure')
#   dir_option = os.path.join(dir_work, 'option')
#   dir_result = os.path.join(dir_work, 'result')
#   dir_model  = os.path.join(dir_work, 'model')
#   now        = datetime.datetime.now()
#   date_stamp = now.strftime('%Y_%m_%d')
#   time_stamp = now.strftime('%H_%M_%S')

#   path_figure = os.path.join(dir_figure, dir_dataset)
#   path_option = os.path.join(dir_option, dir_dataset)
#   path_result = os.path.join(dir_result, dir_dataset)
#   path_model  = os.path.join(dir_model, dir_dataset)

#   date_figure = os.path.join(path_figure, date_stamp)
#   date_option = os.path.join(path_option, date_stamp)
#   date_result = os.path.join(path_result, date_stamp)
#   date_model  = os.path.join(path_model, date_stamp)

#   # file_figure       = os.path.join(date_figure, f'{time_stamp}.png')
#   file_option       = os.path.join(date_option, f'{time_stamp}.ini')
#   file_result       = os.path.join(date_result, f'{time_stamp}.csv')
#   self_file_model   = os.path.join(date_model, f'self.{time_stamp}.pth')
#   energy_file_model = os.path.join(date_model, f'energy.{time_stamp}.pth')
#   unet_file_model = os.path.join(date_model, f'unet.{time_stamp}.pth')
    
#   if not os.path.exists(dir_figure):  os.makedirs(dir_figure)

#   if not os.path.exists(dir_option):  os.makedirs(dir_option)

#   if not os.path.exists(dir_result):  os.makedirs(dir_result)

#   if not os.path.exists(dir_model):  os.makedirs(dir_model)

#   if not os.path.exists(path_figure):  os.makedirs(path_figure)

#   if not os.path.exists(path_option):  os.makedirs(path_option)

#   if not os.path.exists(path_result):  os.makedirs(path_result)

#   if not os.path.exists(path_model):  os.makedirs(path_model)

#   if not os.path.exists(date_figure):  os.makedirs(date_figure)

#   if not os.path.exists(date_option):  os.makedirs(date_option)

#   if not os.path.exists(date_result):  os.makedirs(date_result)

#   if not os.path.exists(date_model):  os.makedirs(date_model)
    
#   # ======================================================================
#   # Evaluation
#   # ======================================================================
#   PSNR = PeakSignalNoiseRatio().to(device)

#   # ======================================================================
#   # Variables for the results 
#   # ======================================================================
#   val_loss_energy_model_mean = np.zeros(number_epoch)
#   val_loss_energy_model_std  = np.zeros(number_epoch)
#   val_psnr_ori_mean          = np.zeros(number_epoch)
#   val_psnr_mean              = np.zeros(number_epoch)
#   val_psnr_std               = np.zeros(number_epoch)
#   val_psnr_langevin_mean     = np.zeros(number_epoch)
#   val_psnr_langevin_std      = np.zeros(number_epoch)
#   val_loss_unet_mean         = np.zeros(number_epoch)

#   # ======================================================================
#   # Training
#   # ======================================================================
#   # torch.autograd.set_detect_anomaly(True)
#   cnt=0
#   for i in range(number_epoch):
#     val_loss_energy_model = list()
#     val_ori               = list()
#     val_psnr              = list()
#     val_psnr_langevin     = list()
#     val_loss_unet         = list()

#     for j, (image, _) in enumerate(iter(dataloader_train)):
#       # -------------------------------------------------------------------
#       # Applied degradation
#       # -------------------------------------------------------------------
#       # if use_unpaired:
#       #   image, image2 = torch.split(image, batch_size)
#       #   image2_noise  = degradation_func(image2)
#       #   image2        = image2.to(device)
#       #   image2_noise  = image2_noise.to(device)

#       # else:
#       image_noise = gaussian_noise(image, sigma_noise)
#       # image_noise = image + sigma_noise * torch.randn_like(image)
#       # if len(buffer) < 1:
#       #   buffer.push(image_noise)
#       image       = image.to(device)
#       image_noise = image_noise.to(device)
      
#       energy.eval()
#       for p in energy.parameters():
#         p.requires_grad = False
        
#       # -------------------------------------------------------------------
#       # Y0 generation identity, gaussian distribution, zero
#       # -------------------------------------------------------------------
#       unet.train()
#       optim_unet.zero_grad()
#       lr_unet_model = sched_optim_unet.get_last_lr()[0]
      
#       prediction = unet(image_noise)
#       # loss_u = unet.compute_loss(prediction, image)
#       loss_u = criterion(prediction, image)
      
#       loss_u.backward()
#       optim_unet.step()
      
#       sched_optim_unet.step()
#       # prediction = generate_Y0(image_noise, 'id')
#       unet.eval()

#       # -------------------------------------------------------------------
#       # Yn generation
#       # -------------------------------------------------------------------

#       # update = image_noise.clone()
#       # update.requires_grad = True
#       # update = update.to(device)
#       prediction_update = SGLD(prediction.detach(), energy, number_step_langevin, list_lr_langevin[i])

#       # prediction_update = energy.update_prediction_langevin(
#       #                                                     prediction.detach(), 
#       #                                                     number_step_langevin,  
#       #                                                     list_lr_langevin[i], 
#       #                                                     regular_data, 
#       #                                                     add_noise, 
#       #                                                     init_noise_decay)
#       # prediction_update = prediction.clone()
#       # -------------------------------------------------------------------
#       # EBM gradient evaluation
#       # -------------------------------------------------------------------
#       for p in energy.parameters():
#         p.requires_grad = True
#       energy = energy.train()
#       optim_energy.zero_grad()
#       # if use_unpaired:
#       #   loss_energy = energy.compute_loss(image_noise, image2, image_noise, prediction_update.detach())
#       # else:
#       neg = energy(prediction_update.detach())
#       pos = energy(image)
#       loss_energy = neg.mean() - pos.mean()
#       loss_energy.backward() 

#       value_psnr_ori      = to_numpy(PSNR(image_noise, image))
#       value_psnr          = to_numpy(PSNR(prediction, image))
#       value_psnr_langevin = to_numpy(PSNR(prediction_update, image))

#       # -------------------------------------------------------------------
#       # Update networks
#       # -------------------------------------------------------------------
#       # clip_grad(energy.parameters(), optim_energy)
#       optim_energy.step()

#       # -------------------------------------------------------------------
#       # Save results for each batch iteration
#       # -------------------------------------------------------------------        
#       val_loss_energy_model.append(loss_energy.item())
#       val_loss_unet.append(loss_u.item())
#       val_ori.append(value_psnr_ori)
#       val_psnr.append(value_psnr)
#       val_psnr_langevin.append(value_psnr_langevin)
        
#     # -------------------------------------------------------------------
#     # Update schedulers
#     # -------------------------------------------------------------------
#     lr_energy_model = sched_optim_energy.get_last_lr()[0]

#     if use_energy_sched:
#       sched_optim_energy.step()

#     # -------------------------------------------------------------------
#     # Save results for each epoch
#     # -------------------------------------------------------------------
#     val_loss_energy_model_mean[i] = np.mean(val_loss_energy_model)
#     val_loss_energy_model_std[i]  = np.std(val_loss_energy_model)
#     val_loss_unet_mean[i]         = np.mean(val_loss_unet)
#     val_psnr_ori_mean[i]          = np.mean(val_ori)
#     val_psnr_mean[i]              = np.mean(val_psnr)
#     val_psnr_std[i]               = np.std(val_psnr)
#     val_psnr_langevin_mean[i]     = np.mean(val_psnr_langevin)
#     val_psnr_langevin_std[i]      = np.std(val_psnr_langevin)

#     log = UNet_Energy_log(number_step_langevin) % (i, 
#                                                     number_epoch, 
#                                                     val_loss_unet_mean[i],
#                                                     val_loss_energy_model_mean[i],
#                                                     lr_unet_model, 
#                                                     lr_energy_model, 
#                                                     val_psnr_ori_mean[i],
#                                                     val_psnr_mean[i], 
#                                                     val_psnr_langevin_mean[i])
#     print(f"{name}:\n", log)
    
#     col = 8
#     if val_psnr_langevin_mean[i] > 27:
#       cnt +=1
#     if i % args.save_plot == 0 or cnt == 1:
#       fig, ax = plt.subplots(4, col, figsize = (3 * 15, 3 * col))
#       for k in range(col):
#         if ngpu > 1:
#           rand = k
#         else:
#           rand = random.randrange(args.batch_size)
#         ax[0][k].set_title('Train (Input)')
#         ax[0][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
#         ax[1][k].set_title(f'Train (Noisy) :{100*sigma_noise}')
#         ax[1][k].imshow(torchvision.utils.make_grid(image_noise[rand].detach().cpu(), normalize=True).permute(1,2,0))
#         ax[2][k].set_title('Train (UNet)')
#         ax[2][k].imshow(torchvision.utils.make_grid(prediction[rand].detach().cpu(), normalize=True).permute(1,2,0))
#         ax[3][k].set_title('Train (Langevin)')
#         ax[3][k].imshow(torchvision.utils.make_grid(prediction_update[rand].detach().cpu(), normalize=True).permute(1,2,0))
#       time_ = datetime.datetime.now().strftime('%HH_%MM')

#       plt.tight_layout()
#       fig.savefig(f'{date_figure}/{name}_log_epoch:{i}_{time_}.png', bbox_inches='tight', dpi=300)
#       plt.close(fig)
#     # TODO: find a condition to break the loop

#     if val_psnr_langevin_mean[i] < val_psnr_langevin_mean[i- int(number_epoch/10)] or cnt == 1:
#     # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
#       print(f'Terminating the run after {i} epochs..')
#       break
    
#     if val_psnr_langevin_mean[i] < 10:
#       print("learning break.. ")
#       break
      
      
#   # ======================================================================
#   # Model evaluation for inferences
#   # ======================================================================
#   energy = energy.eval()
#   unet = unet.eval()

#   # ======================================================================
#   # Training results
#   # ======================================================================
#   (image_train, _) = next(iter(dataloader_train))
#   image_noise_train = gaussian_noise(image_train, sigma_noise)
#   # image_train       = image_train.to(device)
#   image_noise_train = image_noise_train.to(device)
#   y_train           = unet(image_noise_train)
#   # y_train           = image_noise_train.clone()
#   y_update_train    = SGLD(y_train.detach(), energy, number_step_langevin, list_lr_langevin[-1])
#   # y_update_train    = energy.update_prediction_langevin( 
#   #                                                       y_train.detach(), 
#   #                                                       number_step_langevin, 
#   #                                                       list_lr_langevin[-1], 
#   #                                                       regular_data, 
#   #                                                       add_noise, 
#   #                                                       init_noise_decay)

#   # image_train       = to_numpy(permute(image_train)).squeeze()
#   # image_noise_train = to_numpy(permute(image_noise_train)).squeeze()
#   # y_train           = to_numpy(permute(y_train)).squeeze()
#   # y_update_train    = to_numpy(permute(y_update_train)).squeeze()

#   # -------------------------------------------------------------------
#   # Training performance measuring
#   # -------------------------------------------------------------------
#   train_self_data_psnr     = np.zeros(len(dataloader_train))
#   train_energy_data_psnr = np.zeros(len(dataloader_train))

#   for j, (image_train_, _) in enumerate(iter(dataloader_train)):
#     image_noise_train_ = gaussian_noise(image_train_, sigma_noise)
#     image_train_       = image_train_.to(device)
#     image_noise_train_ = image_noise_train_.to(device)
#     y_train_           = unet(image_noise_train_)
#     # y_train_           = image_noise_train_.clone()
#     y_update_train_    = SGLD(y_train_.detach(), energy, number_step_langevin, list_lr_langevin[-1])
#     # y_update_train_    = energy.update_prediction_langevin(
#     #                                                       y_train_.detach(), 
#     #                                                       number_step_langevin, 
#     #                                                       list_lr_langevin[-1], 
#     #                                                       regular_data, 
#     #                                                       add_noise, 
#     #                                                       init_noise_decay)

#     train_self_data_psnr[j]     = to_numpy(PSNR(y_train_.detach(), image_train_)).mean()
#     train_energy_data_psnr[j] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

#   # ======================================================================
#   # Testing results
#   # ======================================================================
#   (image_test, _) = next(iter(dataloader_test))
#   image_noise_test = gaussian_noise(image_test, sigma_noise)
#   image_test       = image_test.to(device)
#   image_noise_test = image_noise_test.to(device)
#   y_test           = unet(image_noise_test)
#   # y_test           = image_noise_test.clone()
#   y_update_test    = SGLD(y_test.detach(), energy, number_step_langevin, list_lr_langevin[-1])
#   # y_update_test    = energy.update_prediction_langevin(
#   #                                                     y_test.detach(), 
#   #                                                     number_step_langevin, 
#   #                                                     list_lr_langevin[-1], 
#   #                                                     regular_data, 
#   #                                                     add_noise, 
#   #                                                     init_noise_decay)

#   # image_test       = to_numpy(permute(image_test)).squeeze()
#   # image_noise_test = to_numpy(permute(image_noise_test)).squeeze()
#   # y_test           = to_numpy(permute(y_test)).squeeze()
#   # y_update_test    = to_numpy(permute(y_update_test)).squeeze()

#   # ---------------------------------------------------clip_grad-------------
#   test_self_data_psnr     = np.zeros(len(dataloader_test))
#   test_energy_data_psnr = np.zeros(len(dataloader_test))

#   for j, (image_test_, _) in enumerate(iter(dataloader_test)):
#     image_noise_test_ = gaussian_noise(image_test_, sigma_noise)
#     # image_test_       = image_test_.to(device)
#     image_noise_test_ = image_noise_test_.to(device)
#     y_test_           = unet(image_noise_test_)
#     # y_test_           = image_noise_test_.clone()
#     y_update_test_    = SGLD(y_test_.detach(), energy, number_step_langevin, list_lr_langevin[-1])
#     # y_test_           = generate_Y0(image_noise_test_, Y0_type)
#     # y_update_test_    = energy.update_prediction_langevin(
#     #                                                       y_test_.detach(), 
#     #                                                       number_step_langevin, 
#     #                                                       list_lr_langevin[-1], 
#     #                                                       regular_data, 
#     #                                                       add_noise, 
#     #                                                       init_noise_decay)

#     test_self_data_psnr[j]     = to_numpy(PSNR(y_test_.detach(), image_test_)).mean()
#     test_energy_data_psnr[j] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

#   # -------------------------------------------------------------------
#   # Save models
#   # -------------------------------------------------------------------
#   torch.save(energy, energy_file_model)
#   torch.save(unet, unet_file_model)
#   # -------------------------------------------------------------------
#   # Save the options
#   # -------------------------------------------------------------------
#   with open(file_option, 'w') as f:
#     f.write('{}: {}\n'.format('work directory', dir_work))
#     f.write('{}: {}\n'.format('cuda device', cuda_device))
#     f.write('{}: {}\n'.format('seed', seed))
#     f.write('{}: {}\n'.format('dataset', dir_dataset))
#     f.write('{}: {}\n'.format('Y0 type', Y0_type))
#     f.write('{}: {}\n'.format('image size', im_size))
#     f.write('{}: {}\n'.format('scale range', scale_range))
#     f.write('{}: {}\n'.format('train size', train_size))
#     f.write('{}: {}\n'.format('test size', test_size))
#     f.write('{}: {}\n'.format('use subset', use_subset))
#     f.write('{}: {}\n'.format('use label', use_label))
#     f.write('{}: {}\n'.format('batch size', batch_size))
#     f.write('{}: {}\n'.format('number epoch', number_epoch))
#     f.write('{}: {}\n'.format('use energy scheduler', use_energy_sched))
#     f.write('{}: {}\n'.format('scheduler step size', sched_step_size))
#     f.write('{}: {}\n'.format('scheduler gamma', sched_gamma))
#     f.write('{}: {}\n'.format('lr energy model', lr_energy_model))
#     f.write('{}: {}\n'.format('lr langevin max', lr_langevin_max))
#     f.write('{}: {}\n'.format('lr langevin min', lr_langevin_min))
#     f.write('{}: {}\n'.format('number step langevin', number_step_langevin))
#     f.write('{}: {}\n'.format('regular data', regular_data))
#     f.write('{}: {}\n'.format('langevin noise decay factor', init_noise_decay))
#     # f.write('{}: {}\n'.format('use energy weights regularization', use_reg))
#     f.write('{}: {}\n'.format('energy weights regularization weight', reg_weight))
#     # f.write('{}: {}\n'.format('use energy L2 weights regularization', use_L2_reg))
#     f.write('{}: {}\n'.format('energy L2 weights regularization weight', L2_reg_weight))
#     f.write('{}: {}\n'.format('add noise', add_noise))
#     f.write('{}: {}\n'.format('use unpaired', use_unpaired))
#     f.write('{}: {}\n'.format('sigma noise', sigma_noise))
#     f.write('{}: {}\n'.format('salt and pepper noise', snp_noise))
#     f.write('{}: {}\n'.format('delete square pixels', square_pixels))
#     f.write('{}: {}\n'.format('degradation', degradation))
#     f.write('{}: {}\n'.format('in channel', in_channel))
#     f.write('{}: {}\n'.format('dim feature', dim_feature))
#     f.write('{}: {}\n'.format('dim output', dim_output))
#     f.write('{}: {}\n'.format('train id psnr mean', np.mean(train_self_data_psnr)))
#     f.write('{}: {}\n'.format('train id psnr std', np.std(train_self_data_psnr)))
#     f.write('{}: {}\n'.format('train energy psnr mean', np.mean(train_energy_data_psnr)))
#     f.write('{}: {}\n'.format('train energy psnr std', np.std(train_energy_data_psnr)))
#     f.write('{}: {}\n'.format('test id psnr mean', np.mean(test_self_data_psnr)))
#     f.write('{}: {}\n'.format('test id psnr std', np.std(test_self_data_psnr)))
#     f.write('{}: {}\n'.format('test energy psnr mean', np.mean(test_energy_data_psnr)))
#     f.write('{}: {}\n'.format('test energy psnr std', np.std(test_energy_data_psnr)))

#   f.close()

#   # -------------------------------------------------------------------
#   # Save the results
#   # -------------------------------------------------------------------   
#   with open(file_result, 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(val_loss_energy_model_mean)
#     writer.writerow(val_loss_energy_model_std)
#     writer.writerow(val_psnr_mean)
#     writer.writerow(val_psnr_std)
#     writer.writerow(val_psnr_langevin_mean)
#     writer.writerow(val_psnr_langevin_std)
#     writer.writerow(train_self_data_psnr)
#     writer.writerow(train_energy_data_psnr)
#     writer.writerow(test_self_data_psnr)
#     writer.writerow(test_energy_data_psnr)

#   f.close()

#   # -------------------------------------------------------------------
#   # Save the figures from training
#   # -------------------------------------------------------------------
#   nRow  = 9
#   nCol  = 6
#   fSize = 3

#   fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))


#   ax[0][0].set_title('UNet Model')
#   ax[0][0].plot(val_loss_unet_mean[:i], color='red', label='Loss')
#   ax[0][0].legend()

#   ax[0][1].set_title('Energy Model')
#   ax[0][1].plot(val_loss_energy_model_mean[:i], color='red', label='Loss')
#   ax[0][1].legend()

#   ax[0][2].set_title('Train PSNR')
#   ax[0][2].plot(val_psnr_mean[:i], color='red', label='ID')
#   ax[0][2].plot(val_psnr_langevin_mean[:i], color='green', label='Langevin')
#   ax[0][2].legend()

#   bplot_colors = ['pink', 'lightgreen']

#   ax[0][3].set_title('ID Data PSNR')
#   ax[0][3].yaxis.grid(True)
#   bplot0 = ax[0][3].boxplot([train_self_data_psnr, 
#                             test_self_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
#   for patch, color in zip(bplot0['boxes'], bplot_colors):
#     patch.set_facecolor(color)

#   ax[0][4].set_title('Energy Data PSNR')
#   ax[0][4].yaxis.grid(True)
#   bplot1 = ax[0][4].boxplot([train_energy_data_psnr, 
#                             test_energy_data_psnr], 0, vert=True, patch_artist=True, labels=['Train', 'Test'])
#   for patch, color in zip(bplot1['boxes'], bplot_colors):
#     patch.set_facecolor(color)
    
#   for k in range(6):
#     if ngpu > 1:
#       rand = k
#     else:
#       rand = random.randrange(args.batch_size)
#     ax[1][k].set_title('Train (Input)')
#     ax[1][k].imshow(torchvision.utils.make_grid(image_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
#     ax[2][k].set_title(f'Train (Noisy): {100*sigma_noise}')
#     ax[2][k].imshow(torchvision.utils.make_grid(image_noise_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
#     ax[3][k].set_title('Train (UNet)')
#     ax[3][k].imshow(torchvision.utils.make_grid(y_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
#     ax[4][k].set_title('Train (Langevin)')
#     ax[4][k].imshow(torchvision.utils.make_grid(y_update_train[rand].detach().cpu(), normalize=True).permute(1,2,0))

#     ax[5][k].set_title('Test (Input)')
#     ax[5][k].imshow(torchvision.utils.make_grid(image_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
#     ax[6][k].set_title(f'Test (Noisy): {100*sigma_noise}')
#     ax[6][k].imshow(torchvision.utils.make_grid(image_noise_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
#     ax[7][k].set_title('Test (UNet)')
#     ax[7][k].imshow(torchvision.utils.make_grid(y_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
    
#     ax[8][k].set_title('Test (Langevin)')
#     ax[8][k].imshow(torchvision.utils.make_grid(y_update_test[rand].detach().cpu(), normalize=True).permute(1,2,0))

#   time_1 = datetime.datetime.now().strftime('%HH_%MM')
#   plt.tight_layout()
#   fig.savefig(f'{date_figure}/last_{name}_{time_1}.png', bbox_inches='tight', dpi=600)
#   plt.close(fig)
  
# if __name__ == '__main__':
#   main()