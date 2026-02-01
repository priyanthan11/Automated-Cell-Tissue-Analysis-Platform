import torch

from models.unet import UNet

model = UNet()
x = torch.randn(1, 1, 256, 256)
y = model(x)

print(f" x Shape: {x.shape}, y Shape: {y.shape}")