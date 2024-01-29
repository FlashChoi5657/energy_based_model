import torch
import torchvision
import numpy as np
import csv, os, argparse, time, random

from torchmetrics import PeakSignalNoiseRatio
from ResNet import IGEBM, EnResnet
from utils import SGLD, get_dataloaders

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="CIFAR10")

parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--dim_feature", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data_size", type=int, default=500)
parser.add_argument("--subset", action="store_true", default=True)

parser.add_argument("--iter_langevin", type=int, default=1500)
parser.add_argument("--lr_langevin_min", type=float, default=0.000005)
parser.add_argument("--lr_langevin_max", type=float, default=0.00001)
parser.add_argument("--noise", type=float, default=0.15)
parser.add_argument("--gpu_devices", type=int, default=0)

args = parser.parse_args()

cuda_device           = args.gpu_devices
NGPU                  = torch.cuda.device_count()
device                = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

energy = EnResnet(args.in_channels, args.dim_feature, activation='swish')
energy = energy.to(device)
sample_x = torch.zeros((args.batch_size, 3, args.img_size, args.img_size))
PSNR = PeakSignalNoiseRatio().to(device)

dir_data = '/hdd1/dataset/'

dataloader_train, dataloader_test, _ = get_dataloaders(dir_data, args.dataset, args.img_size, 
                                                args.batch_size, train_size = args.data_size,
                                                use_subset = args.subset, scale_range=None,
                                                parallel=False)

date_result = '/nas/users/minhyeok/energy_based_model/experiments/result/CIFAR10/2023_05_31'

fSize = 3
col = 8
row = 4
fig, ax = plt.subplots(row, col, figsize = (fSize * 15, fSize * col))
cnt = 0
for i in range(10):
    # init_x = 0.5 + torch.randn_like(sample_x).to(device)
    image, _ = next(iter(dataloader_test))
    image = image.to(device)
    noisy = image + args.noise*torch.randn_like(image)
    
    sample = SGLD(noisy, energy, args.iter_langevin, args.lr_langevin_min)
    psnr = PSNR(noisy, image).detach().cpu().numpy()
    
    if i % 3 == 0:
        for k in range(col):
            rand = random.randrange(args.batch_size)
            ax[cnt][k].imshow(torchvision.utils.make_grid(sample[rand].detach().cpu(), normalize=True).permute(1,2,0))
        cnt += 1
        
    print('batch {}/{} finished, PSNR : {:.5f}'.format((i+1), 10, psnr))

plt.tight_layout()
fig.savefig(f'{date_result}/Annealing.png', bbox_inches='tight', dpi=300)
plt.close(fig)