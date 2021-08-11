import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils

from src.VAE.model_components.Encoder import Encoder
from src.VAE.model_components.Decoder import Decoder
from src.VAE.loss_functions.VAE_loss import VAELoss
from src.VAE.model.VAE_model import VariationalAutoencoder

class Train:

	def __init__(self):
		self.verbose = False
		self.learning_rate = 0.005
		self.epochs = 100
		self.save_model = True

	def eval(self, vae):
		# set to evaluation mode
		vae.eval()

		test_loss_avg, num_batches = 0, 0
		for image_batch, _ in test_dataloader:
		    
			with torch.no_grad():
		    
				image_batch = image_batch.to(device)

				# vae reconstruction
				image_batch_recon, latent_mu, latent_logvar = vae(image_batch.float())

			# reconstruction error
				loss = vae_loss(image_batch_recon, image_batch.float(), latent_mu, latent_logvar)

				test_loss_avg += loss.item()
				num_batches += 1
		    
			test_loss_avg /= num_batches
			print('average reconstruction error: %f' % (test_loss_avg))

	def visualize_results(self, images, model):
		with torch.no_grad():
    
			images = images.to(device)
			images, _, _ = model(images.float())
			images = images.cpu()
			images = to_img(images)
			np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
			plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
			plt.show()

	# This function takes as an input the images to reconstruct
	# and the name of the model with which the reconstructions
	# are performed
	def to_img(x):
		x = x.clamp(0, 1)
		return x

	def show_image(img):
		img = to_img(img)
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))

images, labels = iter(test_dataloader).next()

# First visualise the original images
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()

# Reconstruct and visualise the images using the vae
print('VAE reconstruction:')
visualize_results(images, vae)

    
