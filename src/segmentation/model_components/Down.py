import torch.nn as nn

from src.segmentation.model_components.DoubleConv import DoubleConv

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

