import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from utils.rle import rle_decode
import numpy as np
import os

class CellDatasetRLE(Dataset):
    def __init__(self, img_dir,csv_file):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.image_ids = self.df['ImageId'].unique()
        self.height = self.df['Height'][0]
        self.width = self.df['Width'][0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".png")
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

        # get all RLEs for this image (multiple cells)
        rles = self.df[self.df['ImageId'] == img_id]['EncodedPixels'].values
        mask = np.zeros((self.height, self.width), np.uint8)
        for rle in rles:
            mask += rle_decode(rle,(self.height, self.width))
        mask = np.clip(mask,0,1) # ensure binary

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask
