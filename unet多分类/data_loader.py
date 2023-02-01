import torch.utils.data as Data
import torch
import cv2,os
import glob
# 数据集格式 data/image  data/mask
# image中存放处理好的图像  3通道 rgb格式
# mask中存放标签图像  灰度图  8位  背景为0  第一类1  第二类2 ... 
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
        
        image = image.transpose((2,0,1))
        image = image/255

        label = np.expand_dims(label,0)
        return image, label
    
    def __len__(self):
        return len(self.imgs_path)
