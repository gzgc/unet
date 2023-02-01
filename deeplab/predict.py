
import torch
import numpy as np
import cv2,os
import torch.nn.functional as F
from deeplab import deeplabv3plus

def initialize_net(net_path, device, n_channels, n_classes):
    net = deeplabv3plus()
    net.to(device=device)
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()
    return net


def deal_img(img):
    img = img.transpose((2,0,1))
    img = img/255
    return img


def pred(img,device):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        output = output[-1]
        probs = F.softmax(output, dim=0)
        probs = probs.squeeze(0)

        img_p_class = probs.argmax(dim=0)
        img_p_class = img_p_class.data.cpu().numpy()
    return img_p_class


def set_color(img_p_class,n_class,colors):
    #print(img_p_class.shape)
    img_shape = img_p_class.shape
    h, w = img_shape[0],img_shape[1]
    new_img = np.zeros((h,w,3),np.uint8)
    for i in range(n_class):
        new_img[img_p_class==i] = colors[i]
    return new_img


colors = [[0,0,0],[0,0,255],[0,255,0]]
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model_path = 'best_model.pth'
net = initialize_net(model_path,device,3,3)


if __name__ == '__main__':
    while True:
        img_path = input('data//images//')
        img_path = f'data//images//{img_path}.jpg'
        try:
            img = cv2.imread(img_path)
            img_deal = deal_img(img)
            img_p_class = pred(img=img_deal,device=device)
            color_pred = set_color(img_p_class,3,colors)
            cv2.imshow('colorpre',color_pred)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            continue
    
        

