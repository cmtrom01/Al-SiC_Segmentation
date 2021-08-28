import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, Adadelta, Adagrad, Adamax, ASGD, LBFGS, RMSprop, Rprop
import numpy as np
from tqdm.auto import tqdm as tq

from src.segmentation.models.UNet import UNet
from src.segmentation.loss_functions.BCEDiceLoss import BCEDiceLoss
from src.segmentation.optimizers.RAdam import RAdam
from src.segmentation.data_loader.ImageDataloader import ImageDataloader
from src.segmentation.utils.TrainUtils import TrainUtils
from src.segmentation.utils.Metrics import Metrics

class SegmentationTrainer:

	def __init__(self):
		self.SMOOTH = 1e-6
		self.train_on_gpu = True
		self.train_utils = TrainUtils()
		self.metrics = Metrics()

	def init_model(self, train_on_gpu = True):
		model = UNet(n_channels=3, n_classes=1).float()
		if self.train_on_gpu:
			model.cuda()
		return model

	def init_train_loaders(self):

		DATA_PATH = '/home/chris/Desktop/Al-SiC_Segmentation/data'
		PATH_TRAIN = os.path.join(DATA_PATH, 'images')


		fnames = np.array(os.listdir(PATH_TRAIN))

		num_workers = 2
		bs = 8


		train_dataset = ImageDataloader(
			data_path = DATA_PATH,
			fnames = fnames,
			transforms = None
		)

		train_loader = DataLoader(
		    train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
		)

		valid_loader = train_loader ##change

		return train_loader, valid_loader

	def train(self):
		model = self.init_model(train_on_gpu = True)
		n_epochs = 32
		train_loss_list = []
		valid_loss_list = []
		dice_score_list = []
		lr_rate_list = []
		valid_loss_min = np.Inf

		criterion = BCEDiceLoss(eps=1.0, activation=None)
		optimizer = RAdam(model.parameters(), lr = 0.05)
		#optimizer = RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
		current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)
		train_loader, valid_loader = self.init_train_loaders()

		for epoch in range(1, n_epochs+1):

			train_loss = 0.0
			valid_loss = 0.0
			dice_score = 0.0

			model.train()
			bar = tq(train_loader, postfix={"train_loss":0.0})
			for data, target in bar:
				# move tensors to GPU if CUDA is available
				if self.train_on_gpu:
				    data, target = data.cuda().float(), target.cuda().float()
				# clear the gradients of all optimized variables
				optimizer.zero_grad()
				# forward pass: compute predicted outputs by passing inputs to the model
				output = model(data)
				# calculate the batch loss
				loss = criterion(output, target)
				#print(loss)
				# backward pass: compute gradient of the loss with respect to model parameters
				loss.backward()
				# perform a single optimization step (parameter update)
				optimizer.step()
				# update training loss
				train_loss += loss.item()*data.size(0)
				bar.set_postfix(ordered_dict={"train_loss":loss.item()})
		    ######################    
		    # validate the model #
		    ######################
			model.eval()
			del data, target
			with torch.no_grad():
				bar = tq(valid_loader, postfix={"valid_loss":0.0, "dice_score":0.0})
				for data, target in bar:
			    # move tensors to GPU if CUDA is available
					if self.train_on_gpu:
						data, target = data.cuda().float(), target.cuda().float()
				    # forward pass: compute predicted outputs by passing inputs to the model
					output = model(data)
				    # calculate the batch loss
					loss = criterion(output, target)
				    # update average validation loss 
					valid_loss += loss.item()*data.size(0)
					dice_cof = self.train_utils.dice_no_threshold(output.cpu(), target.cpu()).item()
					dice_score +=  dice_cof * data.size(0)
					bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})
			    
		    # calculate average losses
			train_loss = train_loss/len(train_loader.dataset)
			valid_loss = valid_loss/len(valid_loader.dataset)
			dice_score = dice_score/len(valid_loader.dataset)
			train_loss_list.append(train_loss)
			valid_loss_list.append(valid_loss)
			dice_score_list.append(dice_score)
			lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
		    
		    # print training/validation statistics 
			print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f}'.format(epoch, train_loss, valid_loss, dice_score))
		    
		    # save model if validation loss has decreased
			if valid_loss <= valid_loss_min:
				print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
				torch.save(model.state_dict(), 'model.pt')
				valid_loss_min = valid_loss
		    
			scheduler.step(valid_loss)



