import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.optim as optim


# ======================================================================
# loss 
# ======================================================================
def compute_loss_cross_entropy_positive(prediction_negative, prediction_positive):
    criterion       = nn.BCEWithLogitsLoss()
    label_positive  = torch.ones_like(prediction_positive)
    label_negative  = torch.zeros_like(prediction_negative)        
    loss_positive   = criterion(prediction_positive, label_positive)
    loss_negative   = criterion(prediction_negative, label_negative)
    loss            = 0.5 * (loss_positive + loss_negative) 
    return loss


def compute_loss_cross_entropy_negative(prediction_negative):
    criterion       = nn.BCEWithLogitsLoss()
    label_positive  = torch.ones_like(prediction_negative)        
    loss            = criterion(prediction_negative, label_positive)
    return loss


def compute_loss_contrastive_divergence_positive(prediction_negative, prediction_positive):
    loss = torch.mean(prediction_negative) - torch.mean(prediction_positive)
    return loss


def compute_loss_contrastive_divergence_negative(prediction_negative):
    loss = - torch.mean(prediction_negative)
    return loss


def compute_loss_positive(prediction_negative, prediction_positive, option='cd'):
    if option.lower() == 'cd':
        loss = compute_loss_contrastive_divergence_positive(prediction_negative, prediction_positive)
    elif option.lower() == 'ce':
        loss = compute_loss_cross_entropy_positive(prediction_negative, prediction_positive)
    return loss 


def compute_loss_negative(prediction_negative, option='cd'):
    if option.lower() == 'cd':
        loss = compute_loss_contrastive_divergence_negative(prediction_negative)
    elif option.lower() == 'ce':
        loss = compute_loss_cross_entropy_negative(prediction_negative)
    return loss

 
def compute_gradient(input, prediction, option=1):
    # the two implementations below give the same results
    if option == 1:
        gradient    = torch.autograd.grad(outputs=prediction.sum(), inputs=input, create_graph=True, retain_graph=True)[0]
    
    elif option == 2:
        device      = input.device 
        ones        = torch.ones(prediction.size()).to(device)
        gradient    = torch.autograd.grad(outputs=prediction, inputs=input, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    return gradient 

    
def compute_gradient_penalty(input, prediction, option=1):
    if option == 1:
        gradient        = compute_gradient(input, prediction)
        gradient        = gradient.view(input.shape[0], -1)
        gradient_norm   = torch.norm(gradient, p=2, dim=1)
        penalty         = torch.mean((gradient_norm - 1.0).pow(2))
    
    elif option == 2:  
        gradient        = compute_gradient(input, prediction)
        gradient        = gradient.view(input.shape[0], -1)
        gradient_norm   = torch.norm(gradient, p=2, dim=1)
        penalty         = torch.mean(gradient_norm.pow(2))
    
    return penalty 


def compute_total_variation(data):
    dx = torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:])
    dy = torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :])
    total_variation = torch.mean(dx) + torch.mean(dy)
    return total_variation 

