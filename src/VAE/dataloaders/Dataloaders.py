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
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, 'train', self.fnames[idx])
        mask_path = os.path.join(self.data_path, 'masks', self.fnames[idx]).replace('rw', 'gt_seg_aug-sic')

        st = mask_path.replace('rw', 'gt_seg_aug-sic')[-8:-4]
        stt = str(int('4' + st) - 1)[1:]

        mask_path = mask_path.replace(st, stt)

        
        #img = cv2.cvtColor(cv2.imread(img_path), cv2.IMREAD_COLOR)
        #mask = cv2.imread(mask_path.replace('rw', 'seg'), cv2.IMREAD_GRAYSCALE)

        img, mask = cv2.imread(img_path), cv2.imread(mask_path)

        
        '''
        print('-'*30)
        print(mask)
        print(np.unique(mask))
        print('-'*30)
        '''

       

        
        '''
        if self.preprocess_input:
            # Normalizing the image with the given mean and
            # std corresponding to each channel.
            img = self.preprocess_input(image = img)['image']
        '''
        # PyTorch assumes images in channels-first format. 
        # Hence, bringing the channel at the first place.
        '''The second onehot way'''
        # Palette
        '''
        palette = [[0], [1], [2], [3], [4], [5]]
        
        # Extend the last dimension for one-hot mapping
        mask_onehot2 = np.expand_dims(mask, axis=2)# shape = (H, W) -> (H, W, 1) 
        # one-hot encoding result
        mask = mask_to_onehot(mask_onehot2, palette)  # shape = (H, W, K)
        '''
        
        '''
        if self.transforms:
            # Applying augmentations if any. 
            img, mask = self.transforms(img, mask)
        '''
        


        #mask = bin_image.reshape((512, 256, 6)).transpose((2, 0, 1))
        

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
            mask = RGB_to_Binary_OneHot(1, temp_arr, classes)
       
        

        '''

        print('-'*30)
        print(mask)
        print(np.unique(mask))
        print('-'*30)
        '''

        
        w = 256
        h = 256

        centerx = int(316/2)
        centery = int(531/ 2)
        x = centerx - w/2
        y = centery - h/2

        img = img[int(y):int(y+h), int(x):int(x+w)]
        mask = mask[int(y):int(y+h), int(x):int(x+w)]

        mask = cv2.resize(mask.reshape((256, 256)), (28, 28))
        mask = mask.reshape((28, 28, 1))
        '''
        print('-'*30)
        print(mask)
        print(np.unique(mask))
        print('-'*30)

        print('-'*30)
        print('-'*30)
        print('-'*30)
        print('-'*30)
      
        '''

        img = img / 255.0

        '''
        img2 = np.zeros( ( np.array(img).shape[0], np.array(img).shape[1], 3 ) )
        img2[:,:,0] = img # same value in each channel
        img2[:,:,1] = img
        img2[:,:,2] = img
        '''

        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
            
        return mask, mask
