import torch

from training.metrics import dice_score, iou_score

preds = torch.rand(2, 1, 256, 256)
targets = (torch.rand(2, 1, 256, 256) > 0.5).float()

print("Dice:", dice_score(preds, targets))
print("IoU:", iou_score(preds, targets))
