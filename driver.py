import os
import torch
import numpy as np
import random
from src.segmentation.utils.Train import SegmentationTrainer
from src.VAE.utils.Train import VAETrainer

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    #model_trainer = SegmentationTrainer()
    #train_model = model_trainer.train()
    
    model_trainer = VAETrainer()
    train_model = model_trainer.train()

if __name__ == '__main__':
    main()
