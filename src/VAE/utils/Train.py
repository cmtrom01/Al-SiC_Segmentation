import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.VAE.model_components.Encoder import Encoder
from src.VAE.model_components.Decoder import Decoder
from src.VAE.loss_functions.VAE_loss import VAELoss
from src.VAE.model.VAE_model import VariationalAutoencoder
from src.VAE.dataloaders.VAEDataloader import VAEDataloader

class VAETrainer:

	def __init__(self):
		self.verbose = False
		self.learning_rate = 0.005
		self.epochs = 100
		self.save_model = True
		self.gpu = True
		self.vaeloss = VAELoss()
		self.save_path = '/home/chris/Desktop/Al-SiC_Segmentation/models/BettiVAE'

	def init_model(self):
		vae = VariationalAutoencoder()
		if self.gpu == True:
			vae.cuda()
		return vae

	def init_train_loaders(self):

		DATA_PATH = '/home/chris/Desktop/Al-SiC_Segmentation/data'
		PATH_TRAIN = os.path.join(DATA_PATH, 'images')


		fnames = np.array(os.listdir(PATH_TRAIN))

		num_workers = 2
		bs = 8


		train_dataset = VAEDataloader(
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
		    
		train_dataloader, test_dataloader = self.init_train_loaders()
				
		vae = self.init_model()

		optimizer = torch.optim.Adam(params=vae.parameters(), lr=self.learning_rate, weight_decay=1e-5)

		# set to training mode
		vae.train()

		train_loss_avg = []

		print('Training ...')
		for epoch in range(self.epochs ):
			train_loss_avg.append(0)
			num_batches = 0
		    
			for image_batch, _ in train_dataloader:
			
				image_batch = image_batch.cuda().float()

				image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
			
				loss = self.vaeloss.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
			
				optimizer.zero_grad()
				loss.backward()
			
				optimizer.step()
			
				train_loss_avg[-1] += loss.item()
				num_batches += 1
			
			train_loss_avg[-1] /= num_batches
			print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, self.epochs, train_loss_avg[-1]))
		if self.save_model == True:
			torch.save(vae.state_dict(), self.save_path)
