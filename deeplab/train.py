import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import cv2
import torch.nn.functional as F
import os
import glob
from loss import LovaszLossSoftmax

from deeplab import deeplabv3plus
from mask_loss import MaskCrossEntropyLoss



class Data_Loader(Data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'images/*.jpg'))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks')
        label_path = label_path.replace('jpg', 'png')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path,0)
        #image = image.reshape(3, image.shape[0], image.shape[1])
        #label = label.reshape(1, label.shape[0], label.shape[1])
        image = image.transpose((2,0,1))
        #print(image.shape)
        image = image/255
        #image = np.expand_dims(image,0)
        #label = onehot(label,3)
        #label = label.transpose((2,0,1))
        #print(label.shape)
        #label = np.expand_dims(label,0)
        #label = label.transpose((2,0,1))
        #print(label.shape)
        #print(label[0,200:300,300:400])
        
        return image, label
    
    def __len__(self):
        return len(self.imgs_path)

def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot


def train_net(net, device, data_path, epochs=50, batch_size=10, lr=0.0001):
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
        for image, label in train_loader:
            optimizer.zero_grad()
            
            


            
            #0foreground_pix = (torch.sum(labels_batched!=0).float()+1)/(cfg.DATA_RESCALE**2*cfg.TRAIN_BATCHES)

            image = image.to(device=device, dtype=torch.float32)

            label = label.to(device=device, dtype=torch.long) 
            #print(label.size())
            pred = net(image)
            #print('.....................')
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()
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
    device = torch.device('cuda')
    #device=1
    net = deeplabv3plus()
    net.load_state_dict(torch.load('best_model.pth'))
    net.to(device=device)
    data_path = "data//"
    train_net(net, device, data_path)
