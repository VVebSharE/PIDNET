# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        # self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {0:0,1:128}
                            #   1: ignore_label, 2: ignore_label, 
                            #   3: ignore_label, 4: ignore_label, 
                            #   5: ignore_label, 6: ignore_label, 
                            #   7: 0, 8: 1, 9: ignore_label, 
                            #   10: ignore_label, 11: 2, 12: 3, 
                            #   13: 4, 14: ignore_label, 15: ignore_label, 
                            #   16: ignore_label, 17: 5, 18: ignore_label, 
                            #   19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                            #   25: 12, 26: 13, 27: 14, 28: 15, 
                            #   29: ignore_label, 30: ignore_label, 
                            #   31: 16, 32: 17, 33: 18}
        self.class_weights = torch.FloatTensor([0.3, 1.3]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):


        files = []
        
        if 'train' in self.list_path:
            root_dir="../Dataset/VineNet"
            image_dir = os.path.join(root_dir, 'images')
            mask_dir = os.path.join(root_dir, 'masks')
            image_filenames = os.listdir(image_dir)
            for image_name in image_filenames:
                image_path = os.path.join(image_dir, image_name)
                mask_path = os.path.join(mask_dir, image_name.replace('.png', '_instanceIds.png'))
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "name": name,
                    "label": mask_path
                })

        else:
            root_dir="../ValDataset/VineNet"
            image_dir = os.path.join(root_dir, 'images')
            mask_dir = os.path.join(root_dir, 'masks')
            image_filenames = os.listdir(image_dir)
            for image_name in image_filenames:
                image_path = os.path.join(image_dir, image_name)
                mask_path = os.path.join(mask_dir, image_name.replace('.png', '_instanceIds.png'))
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "label": mask_path,
                    "name": name
                })

        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(item["label"]),
                           cv2.IMREAD_GRAYSCALE)

        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
if(__name__=="__main__"):
    dataset = Cityscapes(root="../data/VineNet", list_path="test.txt")
    print(len(dataset))
    i= list(dataset[0])
    print(i)
    # for j in i:
    #     if(j is):

    #         print(j.shape)

    label = dataset[0][1]
    print(label)
    label = torch.tensor(label)
    print("uniq:",torch.unique(label))
