import os
import time
import datetime, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchmetrics import PeakSignalNoiseRatio

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

# from models import CondEnergyModel, EnergyModel, UNet
# from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256
from sn_gan import Discriminator
from utils import permute, to_numpy, init_weights, UNet_Energy_log, get_dataloaders, SGLD
from utils import gaussian_noise, Self_Energy_log, SampleBuffer, Custom_Dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='single_denoising')
parser.add_argument("--dataset", type=str, default="StreakArtifact")
parser.add_argument("--seed", type=int, default=24)

parser.add_argument("--in_channel", type=int, default=1)
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--dim_feature", type=int, default=256)

parser.add_argument("--sigma_noise", type=float, default=0.4)
parser.add_argument("--sp_noise", type=float, default=0.1)
parser.add_argument("--square_pixels", type=int, default=10)

parser.add_argument("--lr_energy_model", type=float, default=0.00004)
parser.add_argument("--lr_langevin_min", type=float, default=0.000001)
parser.add_argument("--lr_langevin_max", type=float, default=0.00001)
parser.add_argument("--iter_langevin", type=int, default=10)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--save_plot", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--data_size", type=int, default=6000)
parser.add_argument("--subset", action="store_true", default=True)

parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices


def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    ## network
    net = Discriminator(args.in_channel, args.dim_feature, activation='swish')
    
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers * ngpus_per_node)
    net = torch.nn.parallel.DistributedDataParallel(net, broadcast_buffers=False, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)


    print('==> Preparing data..')
    # dir_data = '/hdd1/dataset'
    dir_data    = '/nas/dataset/users/minhyeok/recon'
    
    # transforms_train = transforms.Compose([
    #     torchvision.transforms.Resize(args.img_size),
    #     # transforms.RandomCrop(32, padding=4),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # if args.dataset.lower() == 'cifar10':
    #     dataset_train = CIFAR10(root=dir_data, train=True, download=True, 
    #                     transform=transforms_train)
    #     dataset_test  = CIFAR10(root=dir_data, train=False, download=True, 
    #                         transform=transforms_train)
    # elif args.dataset.lower() == 'celeba':
    #     dataset_train = torchvision.datasets.CelebA(dir_data, transform=transforms_train, 
    #                                                 split='train', download=True)
    #     dataset_test   = torchvision.datasets.CelebA(dir_data, transform=transforms_train, 
    #                                                  split='test', download=True)
    
    # train_loader, test_loader, _ = get_dataloaders(dir_data, args.dataset, args.img_size, 
    #                                             args.batch_size, train_size = args.data_size,
    #                                             use_subset = args.subset, scale_range=None,
    #                                             parallel=True,
    #                                             num_workers=args.num_workers)
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.img_size),
        # torchvision.transforms.RandomCrop(32, padding=4),
        # torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])
    
    train_set = Custom_Dataset(dir_data, transform=transforms_train)
    test_set = Custom_Dataset(dir_data, transform=transforms_train, train=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True,
                              shuffle=(train_sampler is None), num_workers=args.num_workers, 
                              sampler=train_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, drop_last=True,
                             shuffle=(test_sampler is None), num_workers=args.num_workers,
                             sampler=test_sampler)


    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optim_energy = torch.optim.AdamW(net.parameters(), lr=args.lr_energy_model, betas=(args.b1, 0.999))
    sched_optim_energy = torch.optim.lr_scheduler.StepLR(optim_energy, int(args.epochs/6))
    # sched_optim_energy = torch.optim.lr_scheduler.ExponentialLR(optim_energy, gamma=0.97)

    train(net, optim_energy, sched_optim_energy, train_loader, test_loader, args.gpu, args)
            

def train(energy, optim_energy, sched_optim_energy, train_loader, dataloader_test, device, args):
    list_lr_langevin      = np.linspace(args.lr_langevin_max, args.lr_langevin_min, num=args.epochs, endpoint=True)
    PSNR = PeakSignalNoiseRatio().to(device)
    sigmas_np = np.linspace(0.05, 0.8, args.batch_size)
    noise = torch.Tensor(sigmas_np).view((args.batch_size, 1, 1, 1)).to(device)
    
    dir_work              = '/nas/users/minhyeok/energy_based_model/experiments'
    dir_dataset = args.dataset

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
    
    # if not os.path.exists(dir_figure):  os.makedirs(dir_figure)
    # if not os.path.exists(dir_option):  os.makedirs(dir_option)
    # if not os.path.exists(dir_result):  os.makedirs(dir_result)
    # if not os.path.exists(dir_model):  os.makedirs(dir_model)
    # if not os.path.exists(path_figure):  os.makedirs(path_figure)
    # if not os.path.exists(path_option):  os.makedirs(path_option)
    # if not os.path.exists(path_result):  os.makedirs(path_result)
    # if not os.path.exists(path_model):  os.makedirs(path_model)
    
    if not os.path.exists(date_figure):  os.makedirs(date_figure, exist_ok=True)
    if not os.path.exists(date_option):  os.makedirs(date_option, exist_ok=True)
    if not os.path.exists(date_result):  os.makedirs(date_result, exist_ok=True)
    if not os.path.exists(date_model):  os.makedirs(date_model, exist_ok=True)

    val_loss_energy_model_mean = np.zeros(args.epochs)
    val_psnr_ori_mean          = np.zeros(args.epochs)
    val_loss_unet_mean         = np.zeros(args.epochs)
    val_psnr_mean              = np.zeros(args.epochs)
    val_psnr_langevin_mean     = np.zeros(args.epochs)

    for i in range(args.epochs):
        val_loss_energy_model = list()
        val_ori               = list()
        val_psnr              = list()
        val_psnr_langevin     = list()
        val_loss_unet         = list()

        start = time.time()
        for j, (image, streak) in enumerate(iter(train_loader)):
            
            image       = image.to(device)
            # image_noise = image + args.sigma_noise * torch.randn_like(image)
            image_noise = image + noise * torch.randn_like(image).to(device)
            streak = streak.to(device)
            
            energy.eval()
            for p in energy.parameters():
                p.requires_grad = False
            
            noisy = image_noise.clone()
            # noisy.requires_grad = True
            # for _ in range(args.iter_langevin):
            #     noisy.data = noisy.data + 0.5 * np.sqrt(list_lr_langevin[i]) * torch.randn_like(noisy)

            #     loss = -energy(noisy)
            #     loss = loss.sum()
            #     loss.requires_grad_(True)
            #     loss.backward()
                
            #     noisy.data = noisy.data - 0.5 * list_lr_langevin[i] * noisy.grad
                
            #     noisy.data = torch.clamp(noisy.data, *[-1.0 ,1.0])
            #     noisy.grad.detach_()
            #     noisy.grad.zero_()
            # update = noisy.clone()
            prediction_update = SGLD(noisy.detach(), energy, args.iter_langevin, list_lr_langevin[i])
            
            for p in energy.parameters():
                p.requires_grad = True
            energy.train()
            optim_energy.zero_grad()
        
            pos = energy(image)
            neg = energy(prediction_update.detach())
            # loss = criterion(outputs, targets)
            loss = neg.mean() - pos.mean()

            loss.backward()
            optim_energy.step()


            lr_energy_model = sched_optim_energy.get_last_lr()[0]
            # sched_optim_energy.step()

            value_psnr_ori      = (PSNR(image_noise, image)).detach().cpu().numpy()
            value_psnr          = (PSNR(streak, image)).detach().cpu().numpy()
            value_psnr_langevin = (PSNR(prediction_update, image)).detach().cpu().numpy()
        
            
            val_loss_energy_model.append(loss.item())
            # val_loss_unet.append(loss_u.item())
            val_ori.append(value_psnr_ori)
            val_psnr.append(value_psnr)
            val_psnr_langevin.append(value_psnr_langevin)

        val_loss_energy_model_mean[i] = np.mean(val_loss_energy_model)
        val_loss_unet_mean[i]         = np.mean(val_loss_unet)
        val_psnr_ori_mean[i]          = np.mean(val_ori)
        val_psnr_mean[i]              = np.mean(val_psnr)
        val_psnr_langevin_mean[i]     = np.mean(val_psnr_langevin)
        
        time_spent = time.time() - start
        log = Self_Energy_log(args.iter_langevin) % (i, 
                                                  args.epochs, 
                                                  val_loss_energy_model_mean[i],
                                                  lr_energy_model, 
                                                  val_psnr_ori_mean[i],
                                                  val_psnr_mean[i],
                                                  val_psnr_langevin_mean[i],
                                                  time_spent)
        print(f"{args.name}:\n", log)

        col = 8
        if i % args.save_plot == 0:
            energy.eval()
            fig, ax = plt.subplots(4, col, figsize = (3 * 15, 3 * col))
            for k in range(col):
                rand = k
                ax[0][k].set_title('Train (Input)')
                ax[0][k].imshow(torchvision.utils.make_grid(image[rand].detach().cpu(), normalize=True).permute(1,2,0))
                ax[1][k].set_title('Train (Artifact)')
                ax[1][k].imshow(torchvision.utils.make_grid(streak[rand].detach().cpu(), normalize=True).permute(1,2,0))
                ax[2][k].set_title('Train (Langevin)')
                ax[2][k].imshow(torchvision.utils.make_grid(prediction_update[rand].detach().cpu(), normalize=True).permute(1,2,0))
                time_ = datetime.datetime.now().strftime('%HH_%MM')

            plt.tight_layout()
            fig.savefig(f'{date_figure}/{args.name}_log_epoch:{i}_{time_}.png', bbox_inches='tight', dpi=200)
            plt.close(fig)    
            
        if i>20 and val_psnr_langevin_mean[i] < val_psnr_langevin_mean[i-10]:
    # if np.isnan(val_psnr_mean[i]) or ((i > 10) and (val_psnr_mean[i] < 5)) or ((i > 100) and (val_psnr_mean[i] < 10)):
            print(f'Terminating the run after {i} epochs..')
            break
    
    energy = energy.eval()

    (image_train, image_noise_train) = next(iter(train_loader))
    # image_noise_train = gaussian_noise(image_train, args.sigma_noise)
    # image_train       = image_train.to(device)
    y_train = image_noise_train.to(device)
    # y_train           = unet(image_noise_train)
    # y_train           = generate_Y0(image_noise_train, Y0_type)
    y_update_train    = SGLD(y_train.detach(), energy, args.iter_langevin, list_lr_langevin[-1])

    train_self_data_psnr     = np.zeros(len(train_loader))
    train_energy_data_psnr = np.zeros(len(train_loader))

    for j, (image_train_, image_noise_train_) in enumerate(iter(train_loader)):
        # image_noise_train_ = gaussian_noise(image_train_, args.sigma_noise)
        image_train_       = image_train_.to(device)
        y_train_ = image_noise_train_.to(device)
        # y_train_           = unet(image_noise_train_)
        # y_train_           = generate_Y0(image_noise_train_, Y0_type)
        y_update_train_    = SGLD(y_train_.detach(), energy, args.iter_langevin, list_lr_langevin[-1])

        train_self_data_psnr[j]     = to_numpy(PSNR(y_train_, image_train_)).mean()
        train_energy_data_psnr[j] = to_numpy(PSNR(y_update_train_, image_train_)).mean()

    (image_test, image_noise_test) = next(iter(dataloader_test))
    # image_noise_test = gaussian_noise(image_test, args.sigma_noise)
    # image_test       = image_test.to(device)
    y_test = image_noise_test.to(device)
    # y_test           = unet(image_noise_test)
    # y_test           = generate_Y0(image_noise_test, Y0_type)
    y_update_test    = SGLD(y_test.detach(), energy, args.iter_langevin, list_lr_langevin[-1])

    test_self_data_psnr     = np.zeros(len(dataloader_test))
    test_energy_data_psnr = np.zeros(len(dataloader_test))

    for j, (image_test_, image_noise_test_) in enumerate(iter(dataloader_test)):
        # image_noise_test_ = gaussian_noise(image_test_, args.sigma_noise)
        image_test_       = image_test_.to(device)
        y_test_ = image_noise_test_.to(device)
        # y_test_           = unet(image_noise_test_)
        # y_test_           = generate_Y0(image_noise_test_, Y0_type)
        y_update_test_    = SGLD(y_test_.detach(), energy, args.iter_langevin, list_lr_langevin[-1])

        test_self_data_psnr[j]     = to_numpy(PSNR(y_test_, image_test_)).mean()
        test_energy_data_psnr[j] = to_numpy(PSNR(y_update_test_, image_test_)).mean()

    # -------------------------------------------------------------------
    # Save the options
    # -------------------------------------------------------------------
    with open(file_option, 'w') as f:
        f.write('{}: {}\n'.format('work directory', dir_work))
        f.write('{}: {}\n'.format('dataset', dir_dataset))
        f.write('{}: {}\n'.format('image size', args.img_size))
        f.write('{}: {}\n'.format('lr energy model', lr_energy_model))
        f.write('{}: {}\n'.format('train id psnr mean', np.mean(train_self_data_psnr)))
        f.write('{}: {}\n'.format('train id psnr std', np.std(train_self_data_psnr)))
        f.write('{}: {}\n'.format('train energy psnr mean', np.mean(train_energy_data_psnr)))
        f.write('{}: {}\n'.format('train energy psnr std', np.std(train_energy_data_psnr)))
        f.write('{}: {}\n'.format('test id psnr mean', np.mean(test_self_data_psnr)))
        f.write('{}: {}\n'.format('test id psnr std', np.std(test_self_data_psnr)))
        f.write('{}: {}\n'.format('test energy psnr mean', np.mean(test_energy_data_psnr)))
        f.write('{}: {}\n'.format('test energy psnr std', np.std(test_energy_data_psnr)))

        f.close()

    torch.save(energy, energy_file_model)


    # -------------------------------------------------------------------
    # Save the figures from training
    # -------------------------------------------------------------------
    nRow  = 9
    nCol  = 6
    fSize = 3

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    # ax[0][0].set_title('UNet Model')
    # ax[0][0].plot(val_loss_unet_mean[:i], color='red', label='Loss')
    # ax[0][0].legend()

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
        rand = random.randrange(args.batch_size)
        ax[1][k].set_title('Train (Input)')
        ax[1][k].imshow(torchvision.utils.make_grid(image_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
        
        ax[2][k].set_title(f'Train (Noisy): {100* args.sigma_noise}')
        ax[2][k].imshow(torchvision.utils.make_grid(image_noise_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
        
        ax[3][k].set_title('Train (UNet)')
        ax[3][k].imshow(torchvision.utils.make_grid(y_train[rand].detach().cpu(), normalize=True).permute(1,2,0))
        
        ax[4][k].set_title('Train (Langevin)')
        ax[4][k].imshow(torchvision.utils.make_grid(y_update_train[rand].detach().cpu(), normalize=True).permute(1,2,0))

        ax[5][k].set_title('Test (Input)')
        ax[5][k].imshow(torchvision.utils.make_grid(image_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
        
        ax[6][k].set_title(f'Test (Noisy): {100* args.sigma_noise}')
        ax[6][k].imshow(torchvision.utils.make_grid(image_noise_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
        
        ax[7][k].set_title('Test (UNet)')
        ax[7][k].imshow(torchvision.utils.make_grid(y_test[rand].detach().cpu(), normalize=True).permute(1,2,0))
        
        ax[8][k].set_title('Test (Langevin)')
        ax[8][k].imshow(torchvision.utils.make_grid(y_update_test[rand].detach().cpu(), normalize=True).permute(1,2,0))

    time_1 = datetime.datetime.now().strftime('%HH_%MM')
    plt.tight_layout()
    fig.savefig(f'{date_figure}/last_{args.name}_{time_1}.png', bbox_inches='tight', dpi=600)
    plt.close(fig)


if __name__=='__main__':
    main()