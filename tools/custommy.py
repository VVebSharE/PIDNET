# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# color_map = [(128, 64,128),
#              (244, 35,232),
#              ( 70, 70, 70),
#              (102,102,156),
#              (190,153,153),
#              (153,153,153),
#              (250,170, 30),
#              (220,220,  0),
#              (107,142, 35),
#              (152,251,152),
#              ( 70,130,180),
#              (220, 20, 60),
#              (255,  0,  0),
#              (  0,  0,142),
#              (  0,  0, 70),
#              (  0, 60,100),
#              (  0, 80,100),
#              (  0,  0,230),
#              (119, 11, 32)]

color_map = [(128, 64,128),
             (244, 35,232),]

device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='dir for pretrained model', default='/storage/vaibhav/GRAPES/PIDNet/tools/output/cityscapes/pidnet_small_cityscapes/best.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='../samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)
    print("imgs",images_list)
    sv_path = args.r+'outputs/'
    
    model = models.pidnet.get_pred_model(args.a, 2)
    model = load_pretrained(model, args.p)
    model.eval()
    with torch.no_grad():
        for img_path in images_list:
            print(img_path)
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                            cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

            img_tensor = input_transform(img)
            img_tensor = img_tensor.transpose((2, 0, 1)).copy()
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)

            pred = model(img_tensor)
            
            pred = F.interpolate(pred, size=img_tensor.size()[-2:], 
                                mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu()
            print(torch.unique(pred), pred.shape)
            pred = pred.numpy()

            # Create the mask from the predicted labels
            mask_img = np.zeros_like(img).astype(np.uint8)
            for i, color in enumerate(color_map):
                for j in range(3):
                    mask_img[:, :, j][pred == i] = color_map[i][j]
            
            mask_img = Image.fromarray(mask_img)
            
            # Blend the mask with the original image
            blended_img = Image.blend(Image.fromarray(img), mask_img, alpha=0.5)
            
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            print(sv_path, img_name)
            blended_img.save(os.path.join(sv_path, img_name))

            
            
        
        