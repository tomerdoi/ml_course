import os
import torch
from scipy.io import *
import scipy.io as sio
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np


class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.labels = self.load_labels(root_dir)
        self.label_dict = None

    def load_labels(self, root_dir):
        labels_file = os.path.join(root_dir, 'imagelabels.mat')
        labels = sio.loadmat(labels_file)['labels'][0]
        self.label_dict = loadmat(labels_file)
        return labels - 1  # Adjust labels to be zero-indexed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Discard the original label
        label = self.labels[idx]
        return image, label


class FlowerDataLoader:
    def __init__(self, root_dir, batch_size=100):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = FlowerDataset(root_dir=self.root_dir, transform=self.transform)
        self.train_indices, self.validation_indices, self.test_indices = None, None, None
        self.split_data(self.dataset, 0.5, 0.25, 0.25)

    def load_labels(self):
        labels_file = os.path.join(self.root_dir, 'imagelabels.mat')
        labels = sio.loadmat(labels_file)['labels'][0]
        return labels - 1

    def split_data(self, dataset, train_ratio, validation_ratio, test_ratio):
        num_samples = len(dataset)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_split = int(train_ratio * num_samples)
        validation_split = int(validation_ratio * num_samples)
        self.train_indices = indices[:train_split]
        self.validation_indices = indices[train_split:(train_split + validation_split)]
        self.test_indices = indices[(train_split + validation_split):]

    def get_data_loader(self, dataset, indices):
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

    def get_train_data_loader(self):
        return self.get_data_loader(self.dataset, self.train_indices)

    def get_validation_data_loader(self):
        return self.get_data_loader(self.dataset, self.validation_indices)

    def get_test_data_loader(self):
        return self.get_data_loader(self.dataset, self.test_indices)


if __name__ == '__main__':
    # Example usage
    train_folder = '/Users/tomerdoitshman/Desktop/other/D_non_shared/ass3_dataset'
    flower_data_loader = FlowerDataLoader(train_folder, batch_size=100)
    train_loader = flower_data_loader.get_train_data_loader()
    validation_loader = flower_data_loader.get_validation_data_loader()
    test_loader = flower_data_loader.get_test_data_loader()
