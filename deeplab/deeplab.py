import numpy as np
import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import models

from batchnorm import SynchronizedBatchNorm2d

from backbone import build_backbone
from ASPP import ASPP

import resnet_atrous as atrousnet
import xception as xception

def build_backbone(backbone_name, pretrained=True, os=16):
	net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
	return net

class deeplabv3plus(nn.Module):
	def __init__(self):
		super(deeplabv3plus, self).__init__()
		self.backbone = None#build_backbone('xception')		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
			dim_out=256, 
			rate=16//16,
			bn_mom = 0.0003)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
			nn.Conv2d(indim, 48, 1, 1, padding=1//2,bias=True),
			SynchronizedBatchNorm2d(48, momentum=0.0003),
			nn.ReLU(inplace=True),		
			)		
		self.cat_conv = nn.Sequential(
			nn.Conv2d(256+48, 256, 3, 1, padding=1,bias=True),
			SynchronizedBatchNorm2d(256, momentum=0.0003),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
			SynchronizedBatchNorm2d(256, momentum=0.0003),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			)
		self.cls_conv = nn.Conv2d(256, 3, 1, 1, padding=0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m,SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone('res101_atrous', os=16)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)


		feature_shallow = self.shortcut_conv(layers[0])


		feature_cat = torch.cat([feature_aspp,feature_shallow],1)

		result = self.cat_conv(feature_cat)

		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

