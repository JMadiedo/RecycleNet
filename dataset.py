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
        operator = 0
        if (idx > self.original_length-1) :
            new_idx = idx % self.original_length
            operator = math.floor(idx/self.original_length)
            idx = new_idx

        category = self.data_frame.iloc[idx, 1]

        img_name = os.path.join(self.root_dir, "dataset-resized",categories[category],
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)

        toReturn = image
        if operator == 1:
            rotated = rotate(image, angle=45, mode = 'wrap')
            toReturn = np.array(255*rotated, dtype = 'uint8')
        elif operator == 2:
            toReturn = np.fliplr(image).copy()
        elif operator == 3:
            toReturn = np.flipud(image).copy()
        elif operator == 4:
            noisy = random_noise(image,var=0.2**2)
            toReturn = np.array(255*noisy, dtype = 'uint8')

        image = toReturn
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image, category

train_dataset = RecycleDataset("train")
augmented_train_dataset = RecycleDataset("train", True)

val_dataset = RecycleDataset("val")
test_dataset = RecycleDataset("test")
debug_dataset = RecycleDataset("debug")


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

def get_debug_loader(batch_size):
        return torch.utils.data.DataLoader(
            debug_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# if __name__ == '__main__':
#     debug_loader = get_debug_loader(1)
#     for batch_num, (inputs, labels) in enumerate(debug_loader, 1):
#         pass
#         # print("train batch_num: ", batch_num)
#         # print("inputs: ", inputs)
#         # print("labels: ", labels)
