import random, os, glob
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import Tuple
import torch.distributed as dist
from PIL import Image

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        if train:
          self.imgs_data       = glob.glob(self.data_dir + '/train/*/Full*/*.dcm')
          self.noisy_imgs_data = glob.glob(self.data_dir + '/train/*/Low*/*.dcm')
        else:
          self.imgs_data       = glob.glob(self.data_dir + '/test/*/Full*/*.dcm')
          self.noisy_imgs_data = glob.glob(self.data_dir + '/test/*/Low*/*.dcm')
    
    def __getitem__(self, index):  
        # read images in grayscale, then invert them
        full = sitk.ReadImage(self.imgs_data[index])
        full_arr = sitk.GetArrayFromImage(full)
        full_arr = (full_arr - np.min(full_arr)) / (np.max(full_arr) - np.min(full_arr))
        full_arr = full_arr * 255.0
        img = Image.fromarray(np.uint8(full_arr[0]))

        low = sitk.ReadImage(self.noisy_imgs_data[index])
        low_arr = sitk.GetArrayFromImage(low)
        low_arr = (low_arr - np.min(low_arr)) / (np.max(low_arr) - np.min(low_arr))
        low_arr = low_arr * 255.0
        noisy_img = Image.fromarray(np.uint8(low_arr[0]))
    
        if self.transform is not None:            
            img = self.transform(img)             
            noisy_img = self.transform(noisy_img)  

        return img, noisy_img

    def __len__(self):
        return len(self.imgs_data)

      
      
class SampleBuffer:
  def __init__(self, max_samples=10000):
    self.max_samples = max_samples
    self.buffer = []

  def __len__(self):
    return len(self.buffer)

  def push(self, clean, noisy):
    clean = clean.detach().to('cpu')
    noisy = noisy.detach().to('cpu')
    for clean, noisy in zip(clean, noisy):
      self.buffer.append((clean, noisy))

      if len(self.buffer) > self.max_samples:
        self.buffer.pop(0)

  def get(self, n_samples, device='cuda'):
    items = random.choices(self.buffer, k=n_samples)
    image, noise_image = zip(*items)
    
    image = torch.stack(image, 0)
    noise_image = torch.stack(noise_image, 0)
    image = image.to(device)
    noise_image = noise_image.to(device)
    
    return image, noise_image


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size
        
def clip_grad(parameters, optimizer):
  with torch.no_grad():
    for group in optimizer.param_groups:
      for p in group['params']:
        state = optimizer.state[p]

        if 'step' not in state or state['step'] < 1:
            continue

        step = state['step']
        exp_avg_sq = state['exp_avg_sq']
        _, beta2 = group['betas']

        bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
        p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

def FID(img_real: torch.Tensor, img_fake: torch.Tensor, features: int = 64) -> float:
  from torchmetrics.image.fid import FrechetInceptionDistance
  fid = FrechetInceptionDistance(feature=features)
  # generate two slightly overlapping image intensity distributions
  fid.update(img_real, real=True)
  fid.update(img_fake, real=False)

  return fid.compute().item()

def IS(imgs: torch.Tensor) -> Tuple[float, float]:
  from torchmetrics.image.inception import InceptionScore
  inception = InceptionScore()
  # generate some images
  inception.update(imgs)
  s1, s2 = inception.compute()

  return s1.item(), s2.item()

def gaussian_noise(image: torch.Tensor, sigma_noise: float) -> torch.Tensor:
  return image + sigma_noise * torch.randn_like(image)

def sp_noise(image: torch.Tensor, prob: float) -> torch.Tensor:
  img = image.clone()
  _, _, h, w = img.shape

  for i in range(h):
    for j in range(w):
      if random.random() < prob:
        if random.random() < 0.5:
          img[:, :, i, j] = torch.ones(img[:, :, i, j].shape)
        else:
          img[:, :, i, j] = torch.zeros(img[:, :, i, j].shape)

  return img

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
    
def delete_square(image: torch.Tensor, pixels: int) -> torch.Tensor:
  img = image.clone()
  _, _, h, w = img.shape
  
  rh = random.randint(0, h)
  rw = random.randint(0, w)

  sub = round(pixels / 2)
  add = pixels - sub
  
  hmin = max(rh - sub, 0)
  hmax = min(rh + add, h - 1)
  vmin = max(rw - sub, 0)
  vmax = min(rw + add, w - 1)

  img[:, :, hmin:hmax, vmin:vmax] = torch.zeros(img[:, :, hmin:hmax, vmin:vmax].shape)

  return img

def generate_Y0(image: torch.Tensor, Y0_type: str = 'id') -> torch.Tensor:
  if Y0_type == 'id':
    prediction = image.clone()
    
  elif Y0_type == 'random':
    prediction = torch.randn_like(image.clone(), device=image.device)

  elif Y0_type == 'zeros':
    prediction = torch.zeros(image.clone().shape, device=image.device)
  
  return prediction

def permute(tensor: torch.Tensor) -> torch.Tensor:
  return tensor.permute(0, 2, 3, 1)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
  return tensor.detach().cpu().numpy()

def to_image(tensor, index: int = 0) -> torch.Tensor:
  return tensor[index, ...]

def SGLD(update, model, step, lr_rate, noise_decay=1.0):
  update.requires_grad = True
  for _ in range(step):
    update.data = update.data + 0.5 * noise_decay * np.sqrt(lr_rate) * torch.randn_like(update)

    loss = -model(update)
    loss = loss.mean()
    loss.requires_grad_(True)
    loss.backward()
    
    update.data = update.data - 0.5 * lr_rate * update.grad.data
    
    
    # activation
    update.data = torch.clamp(update.data, *[-1.0, 1.0])
    update.grad.detach_()
    update.grad.zero_()

  return update

def get_dataloaders(dir_data: str, dir_dataset: str, im_size: int, 
                    batch_size: bool, train_size: int, transform,
                    test_ratio: float = 0.2, use_subset: bool = False, use_label: int = 4, 
                    scale_range: list = [-1, 1], use_unpaired: bool = False, 
                    parallel: bool = False, num_workers: int = 4) -> Tuple[torch.Tensor, torch.Tensor, str, torchvision.transforms.Lambda]:


  if dir_dataset in ['MNIST', 'CIFAR10']:

    if dir_dataset == 'MNIST':
      dataset_train = torchvision.datasets.MNIST(dir_data, transform=transform, train=True, download=True)
      dataset_test  = torchvision.datasets.MNIST(dir_data, transform=transform, train=False, download=True)

    elif dir_dataset == 'CIFAR10':
      dataset_train         = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=True, download=True)
      dataset_train.data    = np.array(dataset_train.data)
      dataset_train.targets = np.array(dataset_train.targets)
      dataset_test          = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=False, download=True)
      dataset_test.data     = np.array(dataset_test.data)
      dataset_test.targets  = np.array(dataset_test.targets)
      
    if use_subset:

      idx_label_train       = dataset_train.targets == use_label
      dataset_train.data    = dataset_train.data[idx_label_train][:train_size]
      dataset_train.targets = dataset_train.targets[idx_label_train][:train_size]

      idx_label_test        = dataset_test.targets == use_label
      dataset_test.data     = dataset_test.data[idx_label_test][:int(test_ratio * train_size)]
      dataset_test.targets  = dataset_test.targets[idx_label_test][:int(test_ratio * train_size)]
    
    else:
      dataset_train.data    = dataset_train.data[:train_size]
      dataset_train.targets = dataset_train.targets[:train_size]
      dataset_test.data     = dataset_test.data[:int(test_ratio * train_size)]
      dataset_test.targets  = dataset_test.targets[:int(test_ratio * train_size)]
  
  elif dir_dataset == 'celeba':
    
    dataset_train = torchvision.datasets.CelebA(dir_data, transform=transform, split='train', download=True)
    dataset_test   = torchvision.datasets.CelebA(dir_data, transform=transform, split='test', download=True)
    
  elif dir_dataset == 'SVHN':

    dataset_train    = torchvision.datasets.SVHN(dir_data, transform=transform, split='train', download=True)
    dataset_test     = torchvision.datasets.SVHN(dir_data, transform=transform, split='test', download=True)

    if use_subset:
      dataset_train.data = dataset_train.data[:train_size]
      dataset_test.data  = dataset_test.data[:int(test_ratio * train_size)]
      
    else:
      dataset_train.data = dataset_train.data[:train_size]
      dataset_test.data  = dataset_test.data[:int(test_ratio * train_size)]




  train_batch_size = 2 * batch_size if use_unpaired else batch_size
  if parallel:
    train_sampler    = torch.utils.data.distributed.DistributedSampler(dataset_train)
    test_sampler     = torch.utils.data.distributed.DistributedSampler(dataset_test)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=train_batch_size, 
                                                  drop_last=True, shuffle=(train_sampler is None), 
                                                  num_workers=num_workers, sampler=train_sampler)
    dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, 
                                                  drop_last=True, shuffle=(train_sampler is None), 
                                                  num_workers=num_workers, sampler=test_sampler)
  else:
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=train_batch_size, drop_last=True, shuffle=True)
    dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True, shuffle=False)

  return dataloader_train, dataloader_test

def get_unsup_dataloaders(dir_data: str, dir_dataset: str, im_size: int, 
                          in_channel: int, batch_size: int, train_size: int, 
                          train_ratio: float, test_size: int, use_subset: bool, use_label: int, 
                          scale_range: list = [-1, 1], use_unsup_subset: bool = False, GPU: int = 1,
                          use_unpaired: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torchvision.transforms.Lambda]:

  transform = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize([im_size, im_size]),
    torchvision.transforms.ToTensor()
  ])

  if dir_dataset in ['MNIST', 'CIFAR10']:

    if in_channel == 1 and dir_dataset in ['CIFAR10']:
      transform.transforms.append(torchvision.transforms.functional.rgb_to_grayscale)

    if scale_range == [-1, 1]:
      transform.transforms.append(torchvision.transforms.Lambda(lambda t: (t * 2) - 1))
      vis_im_transform = torchvision.transforms.Lambda(lambda t: (t + 1) / 2)
      out_activation   = 'tanh'

    elif scale_range == [0, 1]:
      vis_im_transform = torchvision.transforms.Lambda(lambda t: t)
      out_activation   = 'sigmoid'

    if dir_dataset == 'MNIST':
      dataset_train       = torchvision.datasets.MNIST(dir_data, transform=transform, train=True, download=True)
      dataset_unsup_train = torchvision.datasets.MNIST(dir_data, transform=transform, train=True, download=True)
      dataset_test        = torchvision.datasets.MNIST(dir_data, transform=transform, train=False, download=True)

    elif dir_dataset == 'CIFAR10':
      dataset_train               = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=True, download=True)
      dataset_train.data          = np.array(dataset_train.data)
      dataset_train.targets       = np.array(dataset_train.targets)
      dataset_unsup_train         = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=True, download=True)
      dataset_unsup_train.data    = np.array(dataset_unsup_train.data)
      dataset_unsup_train.targets = np.array(dataset_unsup_train.targets)
      dataset_test                = torchvision.datasets.CIFAR10(dir_data, transform=transform, train=False, download=True)
      dataset_test.data           = np.array(dataset_test.data)
      dataset_test.targets        = np.array(dataset_test.targets)

    if use_subset:

      idx_label_train             = dataset_train.targets == use_label
      dataset_train.data          = dataset_train.data[idx_label_train][:train_size]
      dataset_train.targets       = dataset_train.targets[idx_label_train][:train_size]
      dataset_train.data          = dataset_train.data[:int(train_ratio * dataset_train.targets.shape[0])]
      dataset_train.targets       = dataset_train.targets[:int(train_ratio * dataset_train.targets.shape[0])]

      unsup_label                 = use_unsup_label if use_unsup_subset else use_label
      idx_label_unsup_train       = dataset_unsup_train.targets == unsup_label
      dataset_unsup_train.data    = dataset_unsup_train.data[idx_label_unsup_train][:train_size]
      dataset_unsup_train.targets = dataset_unsup_train.targets[idx_label_unsup_train][:train_size]
      unsup_train_size            = 0 if use_unsup_subset else int(train_ratio * dataset_unsup_train.targets.shape[0])
      dataset_unsup_train.data    = dataset_unsup_train.data[unsup_train_size:]
      dataset_unsup_train.targets = dataset_unsup_train.targets[unsup_train_size:]
      train_size                  = dataset_train.targets.shape[0]

      idx_label_test              = dataset_test.targets == unsup_label
      dataset_test.data           = dataset_test.data[idx_label_test][:test_size]
      dataset_test.targets        = dataset_test.targets[idx_label_test][:test_size]
      test_size                   = dataset_test.targets.shape[0]

    else:
      dataset_train.data          = dataset_train.data[:int(train_ratio * train_size)]
      dataset_train.targets       = dataset_train.targets[:int(train_ratio * train_size)]
      dataset_unsup_train.data    = dataset_unsup_train.data[int(train_ratio * train_size):]
      dataset_unsup_train.targets = dataset_unsup_train.targets[int(train_ratio * train_size):]
      dataset_test.data           = dataset_test.data[:test_size]
      dataset_test.targets        = dataset_test.targets[:test_size]

  elif dir_dataset == 'SVHN':

    if in_channel == 1:
      transform.transforms.append(torchvision.transforms.functional.rgb_to_grayscale)

    if scale_range == [-1, 1]:
      transform.transforms.append(torchvision.transforms.Lambda(lambda t: (t * 2) - 1))
      vis_im_transform = torchvision.transforms.Lambda(lambda t: (t + 1) / 2)
      out_activation   = 'tanh'

    elif scale_range == [0, 1]:
      vis_im_transform = torchvision.transforms.Lambda(lambda t: t)
      out_activation   = 'sigmoid'

    dataset_train       = torchvision.datasets.SVHN(dir_data, transform=transform, split='train', download=True)
    dataset_unsup_train = torchvision.datasets.SVHN(dir_data, transform=transform, split='train', download=True)
    dataset_test        = torchvision.datasets.SVHN(dir_data, transform=transform, split='test', download=True)

    if use_subset:
      
      dataset_train.data       = dataset_train.data[:int(train_ratio * dataset_train.data.shape[0])]
      dataset_unsup_train.data = dataset_unsup_train.data[:int(train_ratio * dataset_train.data.shape[0])]
      unsup_train_size         = 0 if use_unsup_subset else int(train_ratio * dataset_unsup_train.data.shape[0])
      dataset_unsup_train.data = dataset_unsup_train.data[unsup_train_size:]
      dataset_test.data        = dataset_test.data[:test_size]

    else:
      dataset_train.data       = dataset_train.data[:int(train_ratio * train_size)]
      dataset_unsup_train.data = dataset_unsup_train.data[int(train_ratio * train_size):]
      dataset_test.data        = dataset_test.data[:test_size]

  train_batch_size       = 2 * batch_size if use_unpaired else batch_size
  dataloader_train       = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=train_batch_size, drop_last=use_unpaired, shuffle=True, pin_memory=True, num_workers=4*GPU)
  dataloader_unsup_train = torch.utils.data.DataLoader(dataset=dataset_unsup_train, batch_size=train_batch_size, drop_last=use_unpaired, shuffle=True, pin_memory=True, num_workers=4*GPU)
  dataloader_test        = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=4*GPU)

  return dataloader_train, dataloader_unsup_train, dataloader_test, out_activation, vis_im_transform

def VAE_log() -> str:
    log_pattern = '[%4d/%4d]  '
    log_pattern += '[L_vae=%.4f | '
    log_pattern += 'LR_vae=%.4f | '
    log_pattern += 'MSE_vae=%.4f | '
    log_pattern += 'KL_vae=%.4f | '
    log_pattern += 'PSNR_Y0=%.4f]'
    return log_pattern

def VAE_Energy_log(number_step_langevin: int) -> str:
    log_pattern = '[%4d/%4d]  '
    log_pattern += '[L_vae=%.4f | '
    log_pattern += 'LR_vae=%.4f | '
    log_pattern += 'MSE_vae=%.4f | '
    log_pattern += 'KL_vae=%.4f]  '
    log_pattern += '[L_ebm=%.4f | '
    log_pattern += 'LR_ebm=%.4f | '
    log_pattern += 'PSNR_Y0=%.4f | '
    log_pattern += f'PSNR_Y{number_step_langevin}=%.4f]'
    return log_pattern

def UNet_log() -> str:
    log_pattern = '[%4d/%4d]  '
    log_pattern += '[L_unet=%.4f | '
    log_pattern += 'LR_unet=%.4f | '
    log_pattern += 'PSNR_Y0=%.4f]'
    return log_pattern

def UNet_Energy_log(number_step_langevin: int) -> str:
    log_pattern = '[%4d/%4d]  '
    log_pattern += '[L_unet=%.4f | '
    log_pattern += '[L_ebm=%.4f | '
    log_pattern += 'LR_unet=%.5f | '
    log_pattern += 'LR_ebm=%.5f | '
    log_pattern += 'Noisy=%.5f  | '
    log_pattern += 'PSNR_Y0=%.5f | '
    log_pattern += f'PSNR_Y{number_step_langevin}=%.5f]'
    return log_pattern

def Self_Energy_log(number_step_langevin: int) -> str:
    log_pattern = '[%4d/%4d]  '
    log_pattern += '[L_ebm=%.4f | '
    log_pattern += 'LR_ebm=%.7f | '
    log_pattern += 'PSNR_Y0=%.5f | '
    log_pattern += f'PSNR_Y:{number_step_langevin}=%.5f]'
    return log_pattern