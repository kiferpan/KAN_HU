import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class GaussianSmoothBand(nn.Module):
    def __init__(self, bands, kernel_size=5, sigma=1.0):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        kernel = self._gaussian_kernel1d(kernel_size, sigma)
        
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            bias=False
        )
        
        self.conv.weight.data = kernel.view(1, 1, -1)
        self.conv.weight.requires_grad_(False)  

    def _gaussian_kernel1d(self, kernel_size, sigma):
        x = torch.linspace(-sigma*3, sigma*3, kernel_size)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()  
        return kernel

    def forward(self, x):

        batch, band, spatial = x.shape
        x = x.permute(0, 2, 1)  # [batch, spatial, band]
        x = x.reshape(batch*spatial, 1, band) 
        x = self.conv(x)  
        
        x = x.view(batch, spatial, band)
        x = x.permute(0, 2, 1)  # [batch, band, spatial]
        return x

