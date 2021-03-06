import torch
import numpy as np 

class Metrics:

	def __init__(self):
		self.SMOOTH = 1e-6

	def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
	    # You can comment out this line if you are passing tensors of equal shape
	    # But if you are passing output from UNet or something it will most probably
	    # be with the BATCH x 1 x H x W shape
	    labels = labels.int()
	    outputs = torch.nn.Sigmoid(outputs)
	    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
	    
	    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
	    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
	    
	    iou = (intersection + self.SMOOTH) / (union + self.SMOOTH)  # We smooth our devision to avoid 0/0
	    
	    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
	    print(thresholded)
	    
	    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
	    
	    
	# Numpy version
	# Well, it's the same function, so I'm going to omit the comments

	def iou_numpy(self, outputs: np.array, labels: np.array):
	    outputs = outputs.squeeze(1)
	    
	    intersection = (outputs & labels).sum((1, 2))
	    union = (outputs | labels).sum((1, 2))
	    
	    iou = (intersection + SMOOTH) / (union + SMOOTH)
	    
	    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
	    
	    return thresholded  # Or thresholded.mean()
