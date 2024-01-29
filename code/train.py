import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from model import pyramidnet
from models import MyResnet
from ResNet import IGEBM
from sn_gan import Discriminator
import argparse
from utils import SGLD


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--num_worker', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices


def main():
    best_acc = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.Resize(32),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='/hdd1/dataset', train=True, download=True, 
                            transform=transforms_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_worker)

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Making model..')

    # net = IGEBM(3, 32, 1, activation='silu', scale_range=[-1,1])
    # net = MyResnet(3, 32)
    net = Discriminator(3, 128)
    net = nn.DataParallel(net)
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, 
    #                       momentum=0.9, weight_decay=1e-4)
    
    train(net, criterion, optimizer, train_loader, device)
            

def train(net, criterion, optimizer, train_loader, device):
    # net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        image_noise = inputs + 0.15 * torch.randn_like(inputs)
        image_noise = image_noise.to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
            
        noisy = image_noise.clone()
        
        pred = SGLD(noisy, net, 10, 0.5)
        # noisy.requires_grad = True
        # for _ in range(10):
        #     loss = -net(noisy)
        #     loss = loss.sum()
        #     loss.requires_grad_(True)
        #     loss.backward()
            
        #     noisy.data = noisy.data - 0.5 * noisy.grad
            
        #     noisy.data = torch.clamp(noisy.data, *[-1.0 ,1.0])
        #     noisy.grad.detach_()
        #     noisy.grad.zero_()
            
        update = pred.clone()
        
        for p in net.parameters():
            p.requires_grad = True
        
        net.train()
        
        pos = net(inputs)
        neg = net(update.detach())
        # loss = criterion(outputs, targets)
        loss = neg.sum() - pos.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # acc = 100 * correct / total
        
        batch_time = time.time() - start
        
        # if batch_idx % 20 == 0:
        #     print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
        #         batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, batch_time))
    
    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))
    

if __name__=='__main__':
    main()