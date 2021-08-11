import torch
import torch.nn as nn

from src.VAE.model_components.Encoder import Encoder
from src.VAE.model_components.Decoder import Decoder
from src.VAE.loss_functions.VAE_loss import VAELoss
from src.VAE.model.VAE_model import VariationalAutoencoder

class VAE:
    def __init__(self, n_channels, n_classes):
        self.verbose = False

    def get_model(self):
        vae = VariationalAutoencoder()
        print(torch.cuda.is_available())

        device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        vae = vae.to(device)

        num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)
