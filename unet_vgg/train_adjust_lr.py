import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import cv2
import torch.nn.functional as F
import os,time
import glob,json,base64
from loss import LovaszLossSoftmax
from load_data import *

from deeplab.deeplab import deeplabv3plus
from mask_loss import MaskCrossEntropyLoss



def train_net(net, device, data_path, epochs=1, batch_size=10, lr=0.0001):
    net.train()
    dataset = Data_Loader(data_path)
    #dataset = BasicDataset('data//image//','data//mask//')
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = LovaszLossSoftmax()
    #criterion = MaskCrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(ignore_index=255)
    best_loss = float('inf')
    
    
    
    for epoch in range(1,1+epochs):
        print(f'epoch--->:{epoch}')
        kk = 0
        loss_epoch = 0
        batch_num = 0
        for image, label in train_loader:
            optimizer.zero_grad()
            #0foreground_pix = (torch.sum(labels_batched!=0).float()+1)/(cfg.DATA_RESCALE**2*cfg.TRAIN_BATCHES)

            image = image.to(device=device, dtype=torch.float32)

            label = label.to(device=device, dtype=torch.long) 
            #print(label.size())
            pred = net(image)
            #print('.....................')
            loss = criterion(pred, label)
            
            loss_epoch += loss
            batch_num += 1
            
            if kk%5 == 0:
              print('Loss/train', loss.item())
            if loss < best_loss:
                #print('best loss:',loss)
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_for_web02.pth')
            loss.backward()
            optimizer.step()
            kk += 1
        print()
        print('epoch {} loss:',loss_epoch/batch_num)
        print()
        torch.save(net.state_dict(), f'model//{epoch}_model.pth')
def adjust_lr(optimizer, itr, max_itr):
  now_lr = 0.007 * (1 - itr/(max_itr+1)) ** 0.9
  optimizer.param_groups[0]['lr'] = now_lr
  optimizer.param_groups[1]['lr'] = 10*now_lr
  return now_lr

def get_params(model, key):
  for m in model.named_modules():
    if key == '1x':
      if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
        for p in m[1].parameters():
          yield p
    elif key == '10x':
      if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
        for p in m[1].parameters():
          yield p



if __name__ == '__main__':
    print('time begin :',time.localtime())
    device = torch.device('cuda:2')
    net = deeplabv3plus()
    #net.load_state_dict(torch.load('model//10_model.pth'))
    net.to(device=device)
    data_path = "../dataset//water_and_gauge1"
    train_net(net, device, data_path)
    print(time.localtime())
