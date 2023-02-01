import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
import cv2,json,base64
import torch.nn.functional as F
import os,time
import glob
from loss import LovaszLossSoftmax

#from u_net import UNet
#from u_net_mini import UNet
#from u_net_change_down import UNet
from unet_mobile import UNet

from vnet1 import VNet 
from vnet3 import VNet as Vt2 



'''
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
        return len(self.imgs_path)'''
class Data_Loader(Data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_name_list = os.listdir(data_path+'/img')

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        
        img_path = data_path+'/img/'+img_name
        mask_path = data_path+'/mask/'+img_name[:-3]+'png'
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_LINEAR)
        
        img = img.transpose((2,0,1))
        img = img/255
        return img, mask
    
    def __len__(self):
        return len(self.img_name_list)

class Img_json_data(Data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.json_list = os.listdir(data_path)
        
        
    def __getitem__(self, index):
        json_name = self.json_list[index]
        json_path = os.path.join(self.data_path, json_name)
        with open(json_path,'r') as f:
            a = json.loads(f.read())
            
            img_bs64 = a['imageData']
            img = self.decode_bs64img(img_bs64)
            
            shape = img.shape
            mask = np.zeros((shape[0],shape[1]),np.uint8)
            for j in a['shapes']:
                p_list = j['points']
                pol = np.array(p_list)
                label = int(j['label'])
                if label==3:
                    label = 1
                cv2.fillPoly(mask, [pol], color=(label))
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_LINEAR)
        img = img/255
        img = img.transpose((2,0,1))
        mask = np.expand_dims(mask,axis=0)
        return img,mask
        
    def decode_bs64img(self,img_bs64):
        imdata = base64.b64decode(img_bs64)
        im_arr = np.fromstring(imdata,np.uint8)
        img = cv2.imdecode(im_arr,cv2.COLOR_RGB2BGR)
        return img
        
    def __len__(self):
        return len(self.json_list)


def train_net(net, device, data_path, epochs=200, batch_size=10, lr=0.0001):
    print(time.time())
    print('begin......')
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
        print(f'\nepoch--->{epoch}\n')
        print()
        if epoch % 8 == 0:
            lr = lr*0.9
        net.train()
        
        loss_sum = 0
        nnnnn = 0
        
        
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
            
            loss_sum += loss
            nnnnn+=1
            
            
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'unet_mobile_8k_78.pth')
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch}-----> loss_mean:{loss_sum/nnnnn}')
        torch.save(net.state_dict(), f'checkpoint/unet_mobile_epoch_{epoch}.pth')
    print(time.time())
    print('end.....')

if __name__ == '__main__':
    device = torch.device('cuda')
    net = UNet(3, 3)
    #net.load_state_dict(torch.load('checkpoint/epoch_30.pth'))
    net.to(device=device)
    data_path = "../dataset/watergauge_web_bms_56/"
    train_net(net, device, data_path)
    print(time.localtime())