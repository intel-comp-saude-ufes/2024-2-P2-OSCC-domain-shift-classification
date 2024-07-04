import torch
from torchvision.transforms import v2

import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.dataset.custom_dataset import CustomDataset
from src.data.dataset.dataset_interface import DatasetInterface

class ImageNDBDataset(DatasetInterface):
    """
    Dataset for images without patches (original images). It has a train and test dataset that can be accessed by the attributes train_dataset and test_dataset. These datasets can be used in the DataLoader class from PyTorch.
    """
    def __init__(self, path, metadata_path, train_size=0.8, train_transform=None, test_transform=None):
        self.images_path = path
        self.metadata_path = metadata_path
        self.train_transform = train_transform if train_transform is not None else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.test_transform = test_transform if train_transform is not None else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.train_size = train_size
        self.labels_names = {0: 'noncarcinoma', 1: 'carcinoma'}
        self.train, self.test = self._train_test_split()

        self.train_dataset = CustomDataset(self.train[0], self.train[1], transform=self.train_transform)
        self.test_dataset = CustomDataset(self.test[0], self.test[1], transform=self.test_transform)

    def _get_files(self):
        """
        Get files from the path by class carcinoma and non-carcinoma
        """
        images = list(self.images_path.glob('*.png'))
        return images
    
    def _train_test_split(self):
        """
        Split the dataset into train and test
        """
        images = self._get_files()
        metadata= pd.read_csv(self.metadata_path)[["path", "diagnosis"]]

        classes = {"OSCC": "carcinoma", "Leukoplakia with dysplasia": "noncarcinoma", "Leukoplakia without dysplasia": "noncarcinoma"}
        metadata["diagnosis"] = metadata["diagnosis"].map(classes)

        # split into train and test
        train, test = train_test_split(metadata, train_size=self.train_size, stratify=metadata["diagnosis"], random_state=42)

        train_images = [image for image in images if image.parts[-1] in train["path"].values]
        test_images = [image for image in images if image.parts[-1] in test["path"].values]

        train_labels = [1 if 'carcinoma' == diagnosis else 0 for diagnosis in train["diagnosis"].values]
        test_labels = [1 if 'carcinoma' == diagnosis else 0 for diagnosis in test["diagnosis"].values]

        return (train_images, train_labels), (test_images, test_labels)