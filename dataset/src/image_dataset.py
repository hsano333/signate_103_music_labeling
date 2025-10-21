from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
from torchvision import transforms


class ImageDataset(Dataset):
    """MNIST dtaa set"""

    def __init__(self, data, mode, image_data=None, train_label=None):
        self.df = data
        # self.tsfm = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        # )
        self.mode = mode
        self.image_data = image_data
        self.train_label = train_label

        self.label_number = 1

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, tmp_idx):
        if self.mode == "train":
            # idx = self.df[tmp_idx]
            image = self.image_data[tmp_idx]
            label = torch.tensor(self.train_label[tmp_idx])
            # print(f"{tmp_idx=}, {label=}, {image.shape=}")
        else:
            idx = tmp_idx
            label = 0
            image = self.image_data[idx][0]

        return image, label

    def get_column_number(self):
        return 1
        # return self.data.shape[1]

    def get_label_number(self):
        return self.df.shape[0]
