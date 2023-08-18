import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

class VisualEncoder(pl.LightningModule):
    """
    Visual encoder for the diffusion policy.
    """
    def __init__(self):
        super().__init__()
        name = 'resnet18'
        func = getattr(torchvision.models, name)
        self.resnet = func(weights=None)
        # remove the final fully connected layer
        # for resnet18, the output dim should be 512
        self.resnet.fc = torch.nn.Identity()

    def forward(self, x):
        return self.resnet(x)