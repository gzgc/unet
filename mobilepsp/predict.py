from mobile_psp import PSPNet
import torch
import numpy as np
import cv2,os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def deal_img(img): 
    img = img.transpose((2,0,1))
    img = img/255
    return img

device = torch.device('cpu')
def initialize_net():
    net = PSPNet(3, 3)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    return net

def inference_one(net, img, device):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        #print(output)
        #print(output.size())
        output = output[-1]
        probs = F.softmax(output, dim=0)
        
        probs = probs.squeeze(0)        # C x H x W
        
        #print('probs:',probs)
        #errors_sorted, perm = torch.sort(errors, 0, descending=True)
        
        
        tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((512,512)),
                    transforms.ToTensor()
                ]
        )
        masks = []
        for prob in probs:
            prob = tf(prob.cpu())
            mask = prob.squeeze().cpu().numpy()
            mask = mask > 0.5
            masks.append(mask)

    return masks
def set_color(masks,img):
    colors = [(0,0,0),(0,255,0),(255,0,0)]
    w, h = 512,512
    img_mask = np.zeros([h, w, 3], np.uint8)
    for idx in range(0, len(masks)):
        image_idx = (mask[idx] * 255).astype(np.uint8)
        array_img = np.asarray(image_idx)
        img_mask[np.where(array_img==255)] = colors[idx]
    #img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img_mask = cv2.cvtColor(np.asarray(img_mask),cv2.COLOR_RGB2BGR)
    cv2.imshow('mask',img_mask)
    #output = cv2.addWeighted(img, 0.1, img_mask, 0.9, 0)
    #cv2.imwrite(osp.join(args.output, img_name), output)
    #cv2.imshow('out',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = 'best_model.pth'
    #net = eval(cfg.model)(cfg)
    net = initialize_net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.eval()
    while True:
        img_path = input('image path:')
        image = cv2.imread(img_path)
        image = image.transpose((2,0,1))
        image = image/255
        
        '''
        img = Image.open(img_path)
        img = np.array(img)
        img_deal = deal_img(img)'''
        
        
        mask = inference_one(net=net,img=image,device=device)
        set_color(mask,image)




    


