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

device = torch.device('cuda')
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
                    transforms.Resize((512,512)),#720,1280)),
                    transforms.ToTensor()
                ]
        )
        masks = []
        for prob in probs:
            prob = tf(prob.cpu())
            mask = prob.squeeze().cpu().numpy()
            mask = mask > 0.25
            masks.append(mask)

    return masks
def set_color(masks,img):
    colors = [(0,0,0),(0,255,0),(255,0,0)]
    w, h = 512,512#1280,720
    img_mask = np.zeros([h, w, 3], np.uint8)
    for idx in range(0, len(masks)):
        image_idx = (mask[idx] * 255).astype(np.uint8)
        array_img = np.asarray(image_idx)
        img_mask[np.where(array_img==255)] = colors[idx]
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img_mask = cv2.cvtColor(np.asarray(img_mask),cv2.COLOR_RGB2BGR)
    #cv2.imshow('mask',img_mask)
    output = cv2.addWeighted(img, 0.5, img_mask, 0.5, 0)
    #cv2.imwrite(osp.join(args.output, img_name), output)
    #cv2.imshow('out',output)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return output


if __name__ == "__main__":
    model_path = 'best_model.pth'
    #net = eval(cfg.model)(cfg)
    net = initialize_net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.eval()

    cap = cv2.VideoCapture('mv//IMG_1746.MOV')

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('46output.avi',fourcc, 30 ,(1280,720))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('fr',frame)
        
        img = np.zeros((1280,1280,3),dtype=np.uint8)

        img[280:1000,:,:] = frame
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
        #cv2.imshow('im',img)
        
        #img = frame
        img1 = img.transpose((2,0,1))
        img1 = img1/255

        mask = inference_one(net=net,img=img1,device=device)
        img2 = set_color(mask,img)
        img3 = cv2.resize(img2,(1280,1280),interpolation=cv2.INTER_LINEAR)[280:1000,:,:]
        cv2.imshow('test',img3)
        out.write(img3)

        #time.sleep(1)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
