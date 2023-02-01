import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import cv2
import os


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
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False),
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Passthrow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        '''
        self.feature = nn.Sequential(
            nn.Conv2d(n_channels,32,3,1,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            )'''
        self.maxpool = nn.MaxPool2d(2)
        self.meanpool = nn.AvgPool2d(2)
        
        self.convpool = nn.Conv2d(in_channels,in_channels,3,2,padding=1,bias=False)
        self.convpool2 = nn.Conv2d(in_channels,in_channels,2,2,padding=0,bias=False)
        self.convpool3 = nn.Conv2d(in_channels,in_channels,5,2,padding=2,bias=False)
        
        self.br = nn.Sequential(
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True))
        
        self.db = DoubleConv(in_channels*4,out_channels)
        
    def forward(self,x):
        #x = self.feature(x)
        #x1 = x[:,:,::2,::2]
        #x2 = x[:,:,1::2,::2]
        #x3 = x[:,:,::2,1::2]
        #x4 = x[:,:,1::2,1::2]
        x5 = self.maxpool(x)
        x6 = self.convpool(x)
        x7 = self.convpool2(x)
        x8 = self.convpool3(x)
        
        #x9 = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim=1)
        x9 = torch.cat((x5,x6,x7,x8),dim=1)
        x10 = self.br(x9)
        x0 = self.db(x10)
        return x0

'''
class Down1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #assert(out_channels%4==0)
        self.down1 = '''


class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self,x,x1):
        x = self.up(x)
        x2 = torch.cat([x,x1],dim=1)
        return self.conv(x2)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        #self.bone = FirstBone(in_channels,32) # w*h*48  256 * 48
        self.bone = DoubleConv(in_channels,32)

        self.down1 = Passthrow(32,64) # w/2,h/2,64  128*64
        self.down2 = Passthrow(64,128) #w/4,..,128   64*128
        self.down3 = Passthrow(128,256)#             32*256
        self.down4 = Passthrow(256,512)#             16*512

        self.up1 = Up(768,256)
        self.up2 = Up(384,128)
        self.up3 = Up(192,64)
        self.up4 = Up(96,64)
        
        self.outc = OutConv(64, out_channels)
    def forward(self,x):
        x1 = self.bone(x)  # 256   48

        x2 = self.down1(x1) # 128  64   
        x3 = self.down2(x2) #  64  128
        x4 = self.down3(x3) #  32  256
        x5 = self.down4(x4) #  16  512

        
        
        x = self.up1(x5, x4)#  512 256
        x = self.up2(x, x3) #  64  192
        x = self.up3(x, x2) #  128 128
        x = self.up4(x, x1) #  256 96
        logits = self.outc(x)
        return logits
        




















        
