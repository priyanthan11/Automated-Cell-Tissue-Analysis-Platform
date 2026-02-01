import os
import random

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

# from torch.utils.tensorboard import SummaryWriter
from models.unet import UNet
from training.dataset import CellDatasetRLE
from training.losses import DiceBCELoss
from training.metrics import dice_score, iou_score
from utils.visualize import show_sample

# --------------------
# Load Configuration
# --------------------

CONFIG_PATH = "C:/___python/Bio-tech Project/Automated Cell & Tissue Analysis Platform/config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------------
# Apply defaults and validations
# ----------------------------------

dataset_cfg = config.get("dataset", {})
training_cfg = config.get("training", {})
checkpoint_cfg = config.get("checkpoint", {})
repro_cfg = config.get("reproducibility", {})

CSV_FILE = dataset_cfg.get("csv_path", None)
IMG_FILE = dataset_cfg.get("img_path", None)
if CSV_FILE is None or not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV_FILE file not found: {CSV_FILE}")


VAL_SPLIT = dataset_cfg.get("val_split", 0.1)
BATCH_SIZE = training_cfg.get("batch_size", 4)
EPOCHS = training_cfg.get("epochs", 50)
LR = training_cfg.get("lr", 1e-3)
VISUALIZE_EVERY = training_cfg.get("visualize_every", 5)
DEVICE = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = checkpoint_cfg.get("dir", "models/")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = repro_cfg.get("seed", 42)


# --------------------
# Reproducibility
# --------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ------------------------
# Prepare Dataset
# ------------------------
dataset = CellDatasetRLE(IMG_FILE, CSV_FILE)

if len(dataset) == 0:
    raise FileNotFoundError(
        "Dataset is empty. Check your CSV file and image paths")

# Adjust batch size of dataset is smaller
BATCH_SIZE = min(BATCH_SIZE, len(dataset))

val_size = max(1, int(len(dataset)*VAL_SPLIT))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False) if val_size > 0 else None


# ------------------------------------
# Model, Loss, Optimizer, Scheduler
# ------------------------------------

model = UNet(in_ch=1, out_ch=1).to(DEVICE)
criterion = DiceBCELoss()
optimizer = Adam(model.parameters(), lr=LR)

scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

# TensorBoard
# writer = SummaryWriter("runs/cell_segmentation")


# ----------------------------
# Training Loop
# ----------------------------
best_val_dice = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        # --------------------------
        # Edge Case: empty batch
        # --------------------------
        if images.shape[0] == 0:
            continue

        images, masks = images.to(DEVICE), masks.to(DEVICE)

        # --------------------
        # Forward + Backward
        # --------------------
        optimizer.zero_grad()
        outputs = model(images)

        # Edge case: shape mismatch
        if outputs.shape != masks.shape:
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)

    # ----------------------------
    # Validation
    # ----------------------------

    if val_loader is not None:
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)

                if outputs.shape != masks.shape:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(masks, outputs)
                val_iou += iou_score(masks, outputs)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
    else:
        avg_val_loss = avg_val_dice = avg_val_iou = 0.0

    scheduler.step(avg_val_loss)

    # ---------------------------
    #         Logging
    # --------------------------
    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Dice: {avg_val_dice:.4f} "
          f"IoU: {avg_val_iou:.4f}")

    # writer.add_scalar("Loss/train", avg_train_loss, epoch)
    # writer.add_scalar("Loss/val", avg_val_loss, epoch)
    # writer.add_scalar("Metrics/val_dice", avg_val_dice, epoch)
    # writer.add_scalar("Metrics/val_iou", avg_val_iou, epoch)

    # -----------------------------
    # Save best model checkpoint
    # -----------------------------
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_unet.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Saved best model at epoch {epoch} with Dice {best_val_dice:.4f}")

    # -----------------------------
    # Optional visualization
    # -----------------------------
    if epoch % VISUALIZE_EVERY == 0:
        try:
            images, masks = next(iter(val_loader))
            images = images.to(DEVICE)
            outputs = model(images)
            pred_mask = (outputs[0] > 0.5).float().cpu()
            show_sample(images[0].cpu(), pred_mask,
                        title=f"Epoch {epoch} Prediction")
        except Exception as e:
            print(f"Visualization skipped due to: {e}")

print("Training complete!")
