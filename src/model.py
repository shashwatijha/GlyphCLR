import torch
import torch.nn as nn
import torchvision.models as models

class GlyphEncoder(nn.Module):
    def __init__(self, projection_dim=128):
        super(GlyphEncoder, self).__init__()
        
        # 1. Load a standard ResNet18
        # We use 'None' for weights because we want to train from scratch on glyphs
        self.encoder = models.resnet18(weights=None)
        
        # 2. Adjust for Grayscale (ResNet expects 3 channels, we have 1)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. Get the number of features before the final layer
        num_ftrs = self.encoder.fc.in_features
        
        # 4. Replace the classification head with a Projection Head
        # This is the secret sauce for Contrastive Learning (CLR)
        self.encoder.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, projection_dim)
        )

    def forward(self, x):
        return self.encoder(x)