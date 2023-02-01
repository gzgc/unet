import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

#from torchvision import models
#import torch.utils.model_zoo as model_zoo
#import torch.backends.cudnn as cudnn



#mobile net v2
class Block(nn.Module):
    def __init__(self,cfg_list):
        super().__init__()
        self.in_channel,self.hidden_channel,self.out_channel,self.stride2,self.residual = cfg_list

        self.conv1x1_1 = nn.Sequential(nn.Conv2d(self.in_channel,self.hidden_channel,kernel_size=1,padding=0,stride=1),
                                      nn.BatchNorm2d(self.hidden_channel),
                                      nn.ReLU6()
                                      )
        self.conv3x3 = nn.Sequential(nn.Conv2d(self.hidden_channel,self.hidden_channel,groups=self.hidden_channel,kernel_size=3,padding=1,stride=self.stride2),
                                      nn.BatchNorm2d(self.hidden_channel),
                                      nn.ReLU6()
                                      )
        self.conv1x1_2 = nn.Sequential(nn.Conv2d(self.hidden_channel,self.out_channel,kernel_size=1,padding=0,stride=1),
                                      nn.BatchNorm2d(self.out_channel),
                                      nn.ReLU6()
                                      )
    def forward(self,x):
        if self.residual:res = x
        output = self.conv1x1_1(x)
        output = self.conv3x3(output)
        output = self.conv1x1_2(output)
        return output if not self.residual else x+output


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    cfg = [(32,32,16,2,0),(16,64,24,1,0),(24,96,24,1,1),(24,144,32,2,0),(32,192,32,1,1),(32,192,128,2,0)
          ]
    def __init__(self,input_channels,output_channels):
        super(PSPNet,self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(input_channels,32,kernel_size=3,padding=1,stride=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU6(inplace=True))
        self.feature = self._make_layers()  
        
        self.feature6x6 = self._make_stages(128,64,6)
        self.feature3x3 = self._make_stages(128,64,3)
        self.feature2x2 = self._make_stages(128,64,2)
        self.feature1x1 = self._make_stages(128,64,1)
        
        self.up90x90 = nn.UpsamplingBilinear2d(size=(64,64))
        self.up720x720 = nn.UpsamplingBilinear2d(size=(512,512))
        self.Conv2 = nn.Sequential(nn.Conv2d(384,128,kernel_size=3,padding=1,stride=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6(inplace=True))
        self.Conv3 = nn.Sequential(nn.Conv2d(128,32,kernel_size=1,stride=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU6(inplace=True))
        self.outc = OutConv(32,output_channels)
   
    def _make_layers(self):
        layers = []
        for x in self.cfg:
            layers.append(Block(x))
        return nn.Sequential(*layers)
    
    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    def forward(self,x):
        output = self.Conv1(x)
        output = self.feature(output)
        
        feature6x6 = self.feature6x6(output)
        
        f6x6 = self.up90x90(feature6x6)
        f3x3 = self.up90x90(self.feature3x3(output))
        f2x2 = self.up90x90(self.feature2x2(output))
        f1x1 = self.up90x90(self.feature1x1(output))
        feature90 = torch.cat([output, f6x6], dim=1)
        feature90 = torch.cat([feature90, f3x3], dim=1)
        feature90 = torch.cat([feature90, f2x2], dim=1)
        feature90 = torch.cat([feature90, f1x1], dim=1)

        feature720 = self.up720x720(feature90)

        
        output = self.Conv2(feature720)
        output = self.Conv3(output)
        logits = self.outc(output)
        return logits
        pass



