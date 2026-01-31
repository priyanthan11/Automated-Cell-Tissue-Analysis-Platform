import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """
        Combined Dice + Binary Cross-Entropy Loss for cell segmentation
        """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss() # pixel-wise BCE

    def forward(self, inputs, target,smooth=1):
        """
                inputs: predicted mask, shape [B,1,H,W], values [0,1]
                targets: ground truth mask, shape [B,1,H,W], values {0,1}
                """
        # Flatten
        inputs = inputs.view(-1)
        targets = target.view(-1)

        #BCE
        bce_loss = self.bce(inputs, targets)

        #Dice
        intersection = (inputs * targets).sum()
        dice_loss = 1-(2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss