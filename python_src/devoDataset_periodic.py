import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DevoDataset(Dataset):

    def __init__(self, pc_data, age_data, transform=None):
        self.pcs = pc_data
        self.age_radians = age_data * 2 * np.pi
        self.age_sine = np.sin(self.age_radians)
        self.age_cosine = np.cos(self.age_radians)
        self.age_mean = np.mean(age_data)
        self.age_std = np.std(age_data)
        self.age_norm = (age_data - self.age_mean) / self.age_std
        self.transform = transform

    def __len__(self):
        return len(self.age)

    def __getitem__(self, index):
        sample_pcs = self.pcs[index]
        sample_age = self.age_norm[index]
        sample_sine = self.age_sine[index]
        sample_cosine = self.age_cosine[index]
        sample_age_data = np.hstack((sample_age, sample_sine, sample_cosine))
        if self.transform:
            sample_pcs = self.transform(sample_pcs)
            sample_age_data = self.transform(sample_age_data)
        return sample_pcs, sample_age_data


if __name__ == '__main__':
    pass
