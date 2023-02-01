from u_net import UNet
import torch
import numpy as np
import cv2,os
import torch.nn.functional as F

def initialize_net(net_path, device, n_channels, n_classes):
    net = UNet(n_channels=n_channels, n_classes=n_classes)
    net.to(device=device)
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()
    return net

def pre_deal(img):
    h,w = img.shape[0],img.shape[1]
    img = img[:,w-h:w]
    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
    return img

def pred_img(img_path):
    img_org = cv2.imread(img_path)
    img_org = pre_deal(img_org)
    img = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    pred = net(img_tensor)
    pred = np.array(pred.data.cpu()[0])[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    pred  = pred.astype(np.uint8)
    cv2.imshow('pre',pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pred


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model_path = 'model_bms.pth'
net = initialize_net(model_path,device,1,1)


if __name__ == '__main__':
    while True:
        img_path = input('img_path:').strip()
        if os.path.isfile(img_path):
            pred_img(img_path)
        else:
            print('no such file')


