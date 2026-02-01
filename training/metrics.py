def dice_score(preds,targets,threshold=0.5,smooth=1e-6):
    """
        Computes Dice coefficient.

        preds: model output probabilities [B,1,H,W]
        targets: ground truth masks [B,1,H,W]
        """
    # Convert probabilities to binary masks
    preds = (preds > threshold).float()

    preds = preds.view(preds.size(0),-1)
    targets = targets.view(targets.size(0),-1)

    intersection = (preds * targets).sum(dim=1)
    union = (preds + targets).sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Computes Intersection over Union (IoU).
    """

    preds = (preds > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    total = preds.sum(dim=1) + targets.sum(dim=1)
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
