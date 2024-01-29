import random
import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import Tuple


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

def init_weights(model: nn.Module, bias: float = 0.01) -> None:
  if isinstance(model, nn.Conv2d):
    nn.init.xavier_normal_(model.weight)
    model.bias.data.fill_(bias)

def get_dataloaders(dir_data: str, dir_dataset: str, im_size: int, 
                    in_channel: int, batch_size: bool, train_size: int, 
                    train_ratio: float, test_size: int, use_subset: bool, use_label: int, 
                    scale_range: list = [-1, 1], use_unpaired: bool = False) -> Tuple[torch.Tensor, torch.Tensor, str, torchvision.transforms.Lambda]:

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
      train_size            = dataset_train.targets.shape[0]
      dataset_train.data    = dataset_train.data[:int(train_ratio * train_size)]
      dataset_train.targets = dataset_train.targets[:int(train_ratio * train_size)]

      idx_label_test        = dataset_test.targets == use_label
      dataset_test.data     = dataset_test.data[idx_label_test][:test_size]
      dataset_test.targets  = dataset_test.targets[idx_label_test][:test_size]
      test_size             = dataset_test.targets.shape[0]
    
    else:
      dataset_train.data    = dataset_train.data[:int(train_ratio * train_size)]
      dataset_train.targets = dataset_train.targets[:int(train_ratio * train_size)]
      dataset_test.data     = dataset_test.data[:test_size]
      dataset_test.targets  = dataset_test.targets[:test_size]

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

    dataset_train    = torchvision.datasets.SVHN(dir_data, transform=transform, split='train', download=True)
    dataset_test     = torchvision.datasets.SVHN(dir_data, transform=transform, split='test', download=True)

    if use_subset:

      dataset_train.data = dataset_train.data[:int(train_ratio * train_size)]
      dataset_test.data  = dataset_test.data[:test_size]

    else:
      dataset_train.data = dataset_train.data[:int(train_ratio * train_size)]
      dataset_test.data  = dataset_test.data[:test_size]

  train_batch_size = 2 * batch_size if use_unpaired else batch_size
  dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=train_batch_size, drop_last=use_unpaired, shuffle=True, pin_memory=True)
  dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True)

  return dataloader_train, dataloader_test, out_activation, vis_im_transform

def get_unsup_dataloaders(dir_data: str, dir_dataset: str, im_size: int, 
                          in_channel: int, batch_size: bool, train_size: int, 
                          train_ratio: float, test_size: int, use_subset: bool, use_label: int, 
                          scale_range: list = [-1, 1], use_unsup_subset: bool = False, 
                          use_unpaired: bool = False) -> [torch.Tensor, torch.Tensor, torch.Tensor, str, torchvision.transforms.Lambda]:

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
  dataloader_train       = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=train_batch_size, drop_last=use_unpaired, shuffle=True, pin_memory=True)
  dataloader_unsup_train = torch.utils.data.DataLoader(dataset=dataset_unsup_train, batch_size=train_batch_size, drop_last=use_unpaired, shuffle=True, pin_memory=True)
  dataloader_test        = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True)

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
    log_pattern += 'LR_unet=%.4f | '
    log_pattern += '[L_ebm=%.4f | '
    log_pattern += 'LR_ebm=%.4f | '
    log_pattern += 'PSNR_Y0=%.4f | '
    log_pattern += f'PSNR_Y{number_step_langevin}=%.4f]'
    return log_pattern

def Self_Energy_log(number_step_langevin: int) -> str:
    log_pattern = '[%4d/%4d]  '
    log_pattern += '[L_ebm=%.4f | '
    log_pattern += 'LR_ebm=%.4f | '
    log_pattern += 'PSNR_Y0=%.4f | '
    log_pattern += f'PSNR_Y{number_step_langevin}=%.4f]'
    return log_pattern