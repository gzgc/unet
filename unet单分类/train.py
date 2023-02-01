import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import cv2
import torch.nn.functional as F

from loss import LovaszLossSoftmax
from data_loader import Data_Loader
from u_net import UNet


def train_net(net, device, data_path, epochs=1, batch_size=3, lr=0.00001,net_path=None):
    net.train()
    if net_path:
        try:
            net.load_state_dict(torch.load(net_path))
            print(net_path,'加载成功')
        except:
            print('加载失败请检查。。。')
            return
    dataset = Data_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    device = torch.device('cuda')
    net = UNet(n_channels=3, n_classes=1)
    
    net.to(device=device)
    data_path = "data//"
    train_net(net, device, data_path)
