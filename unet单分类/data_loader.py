import torch.utils.data as Data
import torch
import cv2,os
import glob

# 数据集格式 data/image  data/mask
# image中存放处理好的图像  3通道 rgb格式
# label中存放标签图像  灰度图 0 或 255 (也可处理成0 ，1)
class Data_Loader(Data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')
        label_path = label_path.replace('jpg', 'png')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        _,label = cv2.threshold(label,30,255,cv2.THRESH_BINARY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        if label.max() > 1:
            label = label / 255
        return image, label
    def __len__(self):
        return len(self.imgs_path)
