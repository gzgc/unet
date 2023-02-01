import os,json,base64,cv2
import os.path as osp
import logging
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir   = imgs_dir
        self.masks_dir  = masks_dir
        self.scale      = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, img_name)

        img = Image.open(img_path)
        mask_path = mask_path.replace('jpg','png')
        mask = Image.open(mask_path)
        #print('img size:',img.size) img size （pillow.size） 只表示长宽
        #print('mask size:',mask.size)
        assert img.size == mask.size, \
            f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

class Data_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'web_img/*.jpg'))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('web_img', 'web_mask')
        label_path = label_path.replace('jpg', 'png')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path,0)

        image = image.transpose((2,0,1))
        image = image/255
        return {'image': torch.from_numpy(image ), 'mask': torch.from_numpy(label)}
    
    def __len__(self):
        return len(self.imgs_path)

class Img_json_data(Dataset):
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
        #img = np.expand_dims(img,axis=0)
        mask = np.expand_dims(mask,axis=0)
        #return img,mask
        #print(img.shape)
        #print(mask.shape)
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
    def decode_bs64img(self,img_bs64):
        imdata = base64.b64decode(img_bs64)
        im_arr = np.fromstring(imdata,np.uint8)
        img = cv2.imdecode(im_arr,cv2.COLOR_RGB2BGR)
        return img
        
    def __len__(self):
        return len(self.json_list)