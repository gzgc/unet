


import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

from unet import NestedUNet
from unet import UNet
from config import UNetConfig


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cfg = UNetConfig()

def deal_img(img):
    img = img.transpose((2,0,1))
    img = img/255
    return img


def inference_one(net, img, device):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        output = output[-1]
        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)        # C x H x W

        tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((256,256)),
                    transforms.ToTensor()
                ]
        )
        masks = []
        for prob in probs:
            prob = tf(prob.cpu())
            mask = prob.squeeze().cpu().numpy()
            mask = mask > cfg.out_threshold
            masks.append(mask)

    return masks
def set_color(masks,img):
    colors = [(0,0,0),(0,255,0),(255,0,0)]
    w, h = 256,256
    img_mask = np.zeros([h, w, 3], np.uint8)
    for idx in range(0, len(masks)):
        image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
        array_img = np.asarray(image_idx)
        img_mask[np.where(array_img==255)] = colors[idx]
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img_mask = cv2.cvtColor(np.asarray(img_mask),cv2.COLOR_RGB2BGR)
    output = cv2.addWeighted(img, 0.1, img_mask, 0.9, 0)
    #cv2.imwrite(osp.join(args.output, img_name), output)
    cv2.imshow('out',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = 'data/checkpoints/epoch_38.pth'
    net = eval(cfg.model)(cfg)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    while True:
        img_path = input('image path:')
        img_ps = '../dataset//watergauge_web_bms_img56/web_img/'+img_path+'.jpg'
        img = Image.open(img_ps)
        img = np.array(img)
        img_deal = deal_img(img)
        mask = inference_one(net=net,img=img_deal,device=device)
        set_color(mask,img)

            

