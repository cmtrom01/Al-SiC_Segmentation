import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt

from src.segmentation.utils.PreprocessUtils import PreprocessUtils

class VAEDataloader(Dataset):
    def __init__(self, 
                 data_path, 
                 fnames, 
                 preprocess_input = None,
                 transforms = None):
        self.data_path = data_path
        self.fnames = fnames
        self.preprocess_input = preprocess_input
        self.transforms = transforms
        self.preprocess_utils = PreprocessUtils()
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, 'images', self.fnames[idx])
        mask_path = os.path.join(self.data_path, 'masks', self.fnames[idx]).replace('rw', 'gt_seg_aug-sic')

        st = mask_path.replace('rw', 'gt_seg_aug-sic')[-8:-4]
        stt = str(int('4' + st) - 1)[1:]

        mask_path = mask_path.replace(st, stt)

        img, mask = cv2.imread(img_path), cv2.imread(mask_path)

        loadType = 1

        c1 = [148, 236, 121]
        c2 = [255, 255, 255]#[79, 79, 47]
        c3 = [40, 53, 204]
        c4 = [127, 127, 255]
        c5 = [255, 127, 127]
        c6 = [208, 224, 64]
        
        tiffArr = np.array([])

        temp_arr = np.array(mask)
       
        if loadType == 1:
            classes = (c1,c2,c3,c4,c5,c6)
            mask = self.preprocess_utils.RGB_to_Binary_OneHot(1, temp_arr, classes)
        
        w = 256
        h = 256

        centerx = int(316/2)
        centery = int(531/ 2)
        x = centerx - w/2
        y = centery - h/2

        img = img[int(y):int(y+h), int(x):int(x+w)]
        mask = mask[int(y):int(y+h), int(x):int(x+w)]

        #mask = cv2.resize(mask.reshape((256, 256)), (28, 28))
        #mask = mask.reshape((28, 28, 1))

        img = img / 255.0

        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
            
        return mask, mask
