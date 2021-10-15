import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from losses import *
from networks import *
from ds import *
import random
random.seed(0)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim
from torchvision import transforms, utils as tv_utils

def train_UGAC(
    netG_A, netG_B,
    netD_A, netD_B,
    train_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=10,
    init_lr=1e-5,
    ckpt_path='../ckpt/ugac',
    list_of_hp = [1, 0.015, 0.01, 0.001, 1, 0.015, 0.01, 0.001, 0.05, 0.05, 0.01],
):
    netG_A.to(device)
    netG_B.to(device)
    netG_A.type(dtype)
    netG_B.type(dtype)
    ####
    netD_A.to(device)
    netD_B.to(device)
    netD_A.type(dtype)
    netD_B.type(dtype)
    ####
    optimizerG = torch.optim.Adam(list(netG_A.parameters())+list(netG_B.parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(netD_A.parameters())+list(netD_B.parameters()), lr=init_lr)
    optimG_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, num_epochs)
    optimD_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, num_epochs)
    ####
    for eph in range(num_epochs):
        netG_A.train()
        netG_B.train()
        netD_A.train()
        netD_B.train()
        avg_recA_loss, avg_recB_loss = 0, 0
        avg_idtA_loss, avg_idtB_loss = 0, 0
        avg_tot_loss = 0
        print(len(train_loader))
        for i, batch in enumerate(train_loader):
            xA, xB = batch[0].to(device).type(dtype), batch[1].to(device).type(dtype)
            #
            xA_orig = xA.clone().detach()
            fake_A, fake_alpha_A, fake_beta_A = netG_B(xB)
            fake_B, fake_alpha_B, fake_beta_B = netG_A(xA)    
            ####
            rec_A, rec_alpha_A, rec_beta_A = netG_B(fake_B)
            rec_B, rec_alpha_B, rec_beta_B = netG_A(fake_A)
            #compute losses
            
            #first generator
            netD_A.eval()
            netD_B.eval()
            netG_A.train()
            netG_B.train()
            ##
            e1 = list_of_hp[0]*F.l1_loss(rec_A, xA) + list_of_hp[1]*bayeGen_loss(rec_A, rec_alpha_A, rec_beta_A, xA)
            t0, t0_alpha, t0_beta = netG_A(xB)
            e2 = list_of_hp[2]*F.l1_loss(t0, xB) + list_of_hp[3]*bayeGen_loss(t0, t0_alpha, t0_beta, xB)
            e3 = list_of_hp[4]*F.l1_loss(rec_B, xB) + list_of_hp[5]*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
            t0, t0_alpha, t0_beta = netG_B(xA)
            e4 = list_of_hp[6]*F.l1_loss(t0, xA) + list_of_hp[7]*bayeGen_loss(t0, t0_alpha, t0_beta, xA)
            #
            t0 = netD_A(fake_B)
            t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            e5 = list_of_hp[8]*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
            #
            t0 = netD_B(fake_A)
            t2 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            e6 = list_of_hp[9]*F.mse_loss(t2, torch.ones(t2.size()).to(device).type(dtype))
            e7 = list_of_hp[10]*(F.l1_loss(fake_A, xA) + F.l1_loss(fake_B, xB))
            total_loss = e1+e2+e3+e4+e5+e6+e7
            optimizerG.zero_grad()
            total_loss.backward()
            optimizerG.step()
            
            #then discriminator
            netD_A.train()
            netD_B.train()
            ##
            t0 = netD_A(xB)
            pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_real = 1*F.mse_loss(
                pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
            )
            t0 = netD_A(fake_B.detach())
            pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_A_pred = 1*F.mse_loss(
                pred_fake_A, torch.zeros(pred_real_A.size()).to(device).type(dtype)
            )
            loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5
            #
            t0 = netD_B(xA)
            pred_real_B = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_B_real = 1*F.mse_loss(
                pred_real_B, torch.ones(pred_real_B.size()).to(device).type(dtype)
            )
            t0 = netD_B(fake_A.detach())
            pred_fake_B = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
            loss_D_B_pred = 1*F.mse_loss(
                pred_fake_B, torch.zeros(pred_real_B.size()).to(device).type(dtype)
            )
            loss_D_B = (loss_D_B_real + loss_D_B_pred)*0.5
            ##
            loss_D = loss_D_A + loss_D_B
            optimizerD.zero_grad()
            loss_D.backward()
            optimizerD.step()
            
            avg_recA_loss += e1.item()
            avg_recB_loss += e3.item()
            avg_idtA_loss += e4.item()
            avg_idtB_loss += e2.item()
            avg_tot_loss += total_loss.item()
            
        avg_recA_loss /= len(train_loader)
        avg_recB_loss /= len(train_loader)
        avg_idtA_loss /= len(train_loader)
        avg_idtB_loss /= len(train_loader)
        avg_tot_loss /= len(train_loader)
        print(
            'epoch: [{}/{}] | avg_tot_loss: {} | avg_recA_loss: {} | avg_recB_loss: {} \
            | avg_idtA_loss: {} | avg_idtB_loss: {}'.format(
                eph, num_epochs, avg_tot_loss, avg_recA_loss, avg_recB_loss, avg_idtA_loss, avg_idtB_loss
            )
        )
        torch.save(netG_A.state_dict(), ckpt_path+'_eph{}_G_A.pth'.format(eph))
        torch.save(netG_B.state_dict(), ckpt_path+'_eph{}_G_B.pth'.format(eph))
        torch.save(netD_A.state_dict(), ckpt_path+'_eph{}_D_A.pth'.format(eph))
        torch.save(netD_B.state_dict(), ckpt_path+'_eph{}_D_B.pth'.format(eph))
        ##
        optimG_scheduler.step()
        optimD_scheduler.step()

    return netG_A, netG_B, netD_A, netD_B