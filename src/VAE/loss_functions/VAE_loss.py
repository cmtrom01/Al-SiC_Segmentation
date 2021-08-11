import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss:
	def __init__(self):
		self.verbose = False

	def vae_loss(self, recon_x, x, mu, logvar):
            latent_dims = 128
            num_epochs = 100
            batch_size = 64
            capacity = 64
            learning_rate = 1e-3
            variational_beta = 1
            use_gpu = True
	    # recon_x is the probability of a multivariate Bernoulli distribution p.
	    # -log(p(x)) is then the pixel-wise binary cross-entropy.
	    # Averaging or not averaging the binary cross-entropy over all pixels here
	    # is a subtle detail with big effect on training, since it changes the weight
	    # we need to pick for the other loss term by several orders of magnitude.
	    # Not averaging is the direct implementation of the negative log likelihood,
	    # but averaging makes the weight of the other loss term independent of the image resolution.
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
	    
	    # KL-divergence between the prior distribution over latent vectors
	    # (the one we are going to sample from when generating new images)
	    # and the distribution estimated by the generator for the given image.
            kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	    
            return recon_loss + variational_beta * kldivergence
