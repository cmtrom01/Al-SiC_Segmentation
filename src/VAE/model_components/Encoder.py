import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = 64
        latent_dims = 128
        num_epochs = 100
        batch_size = 64
        capacity = 64
        learning_rate = 1e-3
        variational_beta = 1
        use_gpu = True
        input_width = 256
        input_length = 256
        dim_1 = int(input_width/4)
        dim_2 = int(input_length/4)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 128 x 14 128
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 64 x 7 64
        self.fc_mu = nn.Linear(in_features=c*2*dim_1*dim_2, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*dim_1*dim_2, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar
