import os
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.tensorboard import SummaryWriter
from models.unet import UNet
from training.dataset import CellDatasetRLE
from training.losses import DiceBCELoss
from training.metrics import dice_score, iou_score
from utils.visualize import show_sample


#--------------------
# Load Configuration
#--------------------

CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

#----------------------------------
# Apply defaults and validations
#----------------------------------

dataset_cfg = config.get("dataset",{})
training_cfg = config.get("training",{})
checkpoint_cfg = config.get("checkpoint",{})
repro_cfg = config.get("reproducibility",{})

CSV_FILE = dataset_cfg.get("csv_path",None)
if CSV_FILE is None or not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV_FILE file not found: {CSV_FILE}")


VAL_SPLIT = dataset_cfg.get("val_split",0.1)
BATCH_SIZE = training_cfg.get("batch_size",4)



