import os
import math
import pandas as pd
from skimage import io
import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

categories = {
    0: "glass",
    1: "paper",
    2: "plastic",
    3: "metal",
}

class RecycleDataset(Dataset):
    """Load dataset."""
    def __init__(self, type, augmented = False):
        self.root_dir = "data"
        csv = self.root_dir+"/"+type+"-files.csv"
        self.data_frame = pd.read_csv(csv)    
        self.original_length = len(self.data_frame)
        self.augmented = augmented
        
    def __len__(self):
        return self.original_length*5 if self.augmented == True else self.original_length

    def __getitem__(self, idx):
        category = self.data_frame.iloc[idx, 1]

        img_name = os.path.join(self.root_dir, "dataset-resized",categories[category],
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        toReturn = image
        image = toReturn
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image, category

train_dataset = RecycleDataset("train")
augmented_train_dataset = RecycleDataset("train", True)
val_dataset = RecycleDataset("val")
test_dataset = RecycleDataset("test")

def get_train_loader(batch_size, augmented = False):
    dataset = augmented_train_dataset if augmented == True else train_dataset
    return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def get_val_loader(batch_size):
        return torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def get_test_loader(batch_size):
        return torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

