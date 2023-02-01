import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import cv2
import os

'''
class Data_Loader(Data):
    def __init__(self):
        pass
    def __len__(self):
        ...
    def get_item(self):
        ...'''

class FirstBone(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.bone = nn.Sequential(
            nn.Conv2d(in_channels,out_channels//2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels//2,out_channels//4,1,bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels//4,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self,x):
        return self.bone(x)


class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels,int(out_channels),kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=2,stride=2,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2,out_channels//2,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        x1 = self.down1(x)
        x2 = self.down(x)
        x = torch.cat((x1,x2),dim=1)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels,out_channels//2,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    def forward(self,x,x1):
        x = self.up(x)
        return torch.cat([x,x1],dim=1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.bone = FirstBone(in_channels,48) # w*h*48  256 * 48

        self.down1 = Down(48,64) # w/2,h/2,64  128*64
        self.down2 = Down(64,96) #w/4,..,128  64*96
        self.down3 = Down(96,128)#             32*128
        self.down4 = Down(128,192)#            16*192

        self.up1 = Up(192,128)
        self.up2 = Up(256,96)
        self.up3 = Up(192,64)
        self.up4 = Up(128,48)
        
        self.outc = OutConv(96, out_channels)
    def forward(self,x):
        x1 = self.bone(x)  # 256   48

        x2 = self.down1(x1) # 128  64
        x3 = self.down2(x2) #  64  96
        x4 = self.down3(x3) #  32  128
        x5 = self.down4(x4) #  16  192
        
        x = self.up1(x5, x4)#  32  256
        x = self.up2(x, x3) #  64  192
        x = self.up3(x, x2) #  128 128
        x = self.up4(x, x1) #  256 96
        logits = self.outc(x)
        return logits
        




















        
