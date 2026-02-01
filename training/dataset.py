import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.rle import rle_decode


class CellDatasetRLE(Dataset):
    def __init__(self, img_dir,csv_file,target_size=(256,256)):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.image_ids = self.df['ImageId'].unique()
        self.height = self.df['Height'][0]
        self.width = self.df['Width'][0]
        self.target_size = target_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # img_id = self.image_ids[idx]
        # img_path = os.path.join(self.img_dir, img_id + ".png")
        # image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        #
        # # get all RLEs for this image (multiple cells)
        # rles = self.df[self.df['ImageId'] == img_id]['EncodedPixels'].values
        # mask = np.zeros((self.height, self.width), np.uint8)
        # for rle in rles:
        #     mask += rle_decode(rle,(self.height, self.width))
        # mask = np.clip(mask,0,1) # ensure binary
        #
        # image = torch.from_numpy(image).unsqueeze(0).float()
        # mask = torch.from_numpy(mask).unsqueeze(0).float()
        # return image, mask
        row = self.df.iloc[idx]

        image_id = row["ImageId"]
        height = int(row["Height"])
        width = int(row["Width"])

        # Load image
        if self.img_dir:
            img_path = os.path.join(self.img_dir, image_id + ".png")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
        else:
            image = np.zeros((height, width), dtype=np.uint8)

        # Decode RLE mask
        mask = rle_decode(row["EncodedPixels"], (height, width))

        # ----------------------------
        # Resize (CRITICAL FIX)
        # ----------------------------
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        # Add channel dimension
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
