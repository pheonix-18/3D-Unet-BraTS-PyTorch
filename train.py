import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import nibabel as nib
import pickle
from tqdm import tqdm
import random
from torchvision.transforms import transforms
from scipy import ndimage
import matplotlib.pyplot as plt
from dataset import BraTS
from model import Unet3D
from criterion import softmax_dice
import torch
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_batch(model, data, optimizer, criterion):
    model.train()
    images, targets = data
    preds = model(images)
    optimizer.zero_grad()
    loss, dice1, dice2, dice3 = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item(), dice1.item(), dice2.item(), dice3.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    images, targets = data
    preds = model(images)
    loss, dice1, dice2, dice3 = criterion(preds, targets)
    return loss.item(), dice1.item(), dice2.item(), dice3.item()

print("Loading Data : ")

train_dataset = BraTS("./train_pkl_all/",'train')
val_dataset = BraTS("./val_pkl_all/",'valid')

train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 2, shuffle = True)

print(f"Train Loader {len(train_loader)} : Val Loader {len(val_loader)}")
model = Unet3D(4, 4, 64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = softmax_dice()

nEpochs = 10
log_interval = 50

with open('logger1.txt', 'w') as f:
    
    for epoch in tqdm(range(nEpochs)):
        train_loss = val_loss = 0
        dice_1_t = dice_2_t= dice_3_t = dice_1_v = dice_2_v = dice_3_v = 0
        for i, data in enumerate(train_loader):
            loss, dice1, dice2, dice3 = train_batch(model, data, opt, criterion)
            train_loss += loss
            dice_1_t += dice1
            dice_2_t += dice2
            dice_3_t += dice3
            if i%log_interval==0:
                print(f"Train Epoch: {epoch}, Iter: {i} Overall Loss: {loss} | L1 Dice : {dice1} | L2 Dice : {dice2} | L3 Dice : {dice3}")
        d = len(train_loader)
        avg_train = f"Train Epoch: {epoch}, Overall Loss: {train_loss/d} | L1 Dice : {dice_1_t/d} | L2 Dice : {dice_2_t/d} | L3 Dice : {dice_3_t/d}"
        print(avg_train)
        f.writelines(avg_train)
        # for i, data in enumerate(val_loader):
        #     loss, dice1, dice2, dice3 = validate_batch(model, data, criterion)
        #     if i%log_interval==0:
        #         print(f"Val Epoch: {epoch}, Iter: {i} Overall Loss: {loss} | L1 Dice : {dice1} | L2 Dice : {dice2} | L3 Dice : {dice3}")
        #     val_loss += loss
        #     dice_1_v += dice1
        #     dice_2_v += dice2
        #     dice_3_v += dice3
        # print('='*50)
        # d = len(val_loader)
        # avg_val = f"Train Epoch: {epoch}, Overall Loss: {val_loss/d} | L1 Dice : {dice_1_v/d} | L2 Dice : {dice_2_v/d} | L3 Dice : {dice_3_v/d}"
        # print(avg_val)
         # f.writelines(avg_val)
        file_name = 'model_f'+str(epoch)+".pth"
        print("Saving Model")
        torch.save(model.state_dict(), file_name)
    





