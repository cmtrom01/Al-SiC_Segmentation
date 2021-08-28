import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.capacity = c = 64
        latent_dims = 128
        num_epochs = 100
        batch_size = 64
        capacity = 64
        learning_rate = 1e-3
        variational_beta = 1
        use_gpu = True
        input_width = 256
        input_length = 256
        self.dim_1 = int(input_width/4)
        self.dim_2 = int(input_length/4)

        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*self.dim_1*self.dim_2)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.capacity*2, self.dim_1, self.dim_2) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
