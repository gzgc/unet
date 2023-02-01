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

from mobile_psp import PSPNet




class Data_Loader(Data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'mask')
        label_path = label_path.replace('jpg', 'png')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path,0)
        #image = image.reshape(3, image.shape[0], image.shape[1])
        #label = label.reshape(1, label.shape[0], label.shape[1])
        image = image.transpose((2,0,1))
        #print(image.shape)
        image = image/255
        #image = np.expand_dims(image,0)
        label = np.expand_dims(label,0)
        #label = label.transpose((2,0,1))
        #print(label.shape)
        #print(label[0,200:300,300:400])
        return image, label
    
    def __len__(self):
        return len(self.imgs_path)




def train_net(net, device, data_path, epochs=1, batch_size=3, lr=0.00001):
    dataset = Data_Loader(data_path)
    #dataset = BasicDataset('data//image//','data//mask//')
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = LovaszLossSoftmax()
    best_loss = float('inf')
    for epoch in range(epochs):

        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            #print(image.size()) 3 3 512 512
            label = label.to(device=device, dtype=torch.float32) 
            
            '''
            label = label.to(device=device, dtype=torch.long)
            '''
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
    net = PSPNet(3,3)
    #net.load_state_dict(torch.load('best_model.pth'))
    net.to(device=device)
    data_path = "data//"
    train_net(net, device, data_path)
