import torch
from torch import nn
from torchvision.transforms import v2

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset, loading image from disk so it is not necessary to load all images in memory
    """	
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        
        labels = self.labels[idx]
        return image, labels
        
class PatchDataset:
    """
    Dataset for patches images. Arranges the patches in train and test sets considering the parent image (prefix of the image name). It has a train and test dataset that can be accessed by the attributes train_dataset and test_dataset.
    These datasets can be used in the DataLoader class from PyTorch.
    """	
    def __init__(self, path, train_size=0.8, transform=None):
        self.path = path
        self.train_size = train_size
        self.transform = transform if transform is not None else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.labels_names = {0: 'noncarcinoma', 1: 'carcinoma'}
        self.train, self.test = self._train_test_split()

        self.train_dataset = CustomDataset(self.train[0], self.train[1], transform=self.transform)
        self.test_dataset = CustomDataset(self.test[0], self.test[1], transform=self.transform)
    
    def _get_files(self):
        """
        Get files from the path by class carcinoma and non-carcinoma
        """
        # carcinoma images
        carcinoma = list(self.path.glob('carcinoma/*.png'))

        # non-carcinoma images
        noncarcinoma = list(self.path.glob('no_dysplasia/*.png'))
        noncarcinoma.extend(list(self.path.glob('with_dysplasia/*.png')))

        return carcinoma, noncarcinoma
    
    def _get_parent_image_name(self, image_name):
        """
        Get the parent image name from the image name
        """
        image_filename = image_name.parts[-1]
        image_root = "".join(image_name.parts[0:-1]) + "_".join(image_filename.split('_')[0:-1])
        return image_root

    def _count_images_parent(self, image_class, image_parent):
        """
        This is a helper function to count the amount of images by parent image name
        """
        amount = {}
        for image in image_class:
            image_name = image.parts[-1]
            for parent in image_parent:
                if parent in image_name:
                    amount[parent] = amount.get(parent, 0) + 1
                    break

        return amount

    def _train_test_split(self):
        """
        Split the dataset into train and test
        """
        carcinoma, noncarcinoma = self._get_files()
        carcinoma_parent = list(set([self._get_parent_image_name(image) for image in carcinoma]))
        noncarcinoma_parent = list(set([self._get_parent_image_name(image) for image in noncarcinoma]))

        # create arrays of 1 and 0 for carcinoma and non-carcinoma
        carcinoma_labels = list(np.ones(len(carcinoma_parent)))
        noncarcinoma_labels = list(np.zeros(len(noncarcinoma_parent)))

        # extend the labels
        labels = carcinoma_labels + noncarcinoma_labels
        images = carcinoma_parent + noncarcinoma_parent

        # split the dataset by parent name
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, train_size=self.train_size, stratify=labels, random_state=42)
        
        # get patches images
        images_patches = carcinoma + noncarcinoma
        
        images_train = [image for image in images_patches if self._get_parent_image_name(image) in images_train]
        labels_train = [1 if 'carcinoma' in str(image) else 0 for image in images_train]

        images_test = [image for image in images_patches if self._get_parent_image_name(image) in images_test]
        labels_test = [1 if 'carcinoma' in str(image) else 0 for image in images_test]

        return (images_train, labels_train), (images_test, labels_test)

    def __len__(self):
        return len(self.train[0]) + len(self.test[0])
    

class ImageDataset:
    """
    Dataset for images without patches (original images). It has a train and test dataset that can be accessed by the attributes train_dataset and test_dataset. These datasets can be used in the DataLoader class from PyTorch.
    """
    def __init__(self, images_path, metadata_path, train_size=0.8, transform=None):
        self.images_path = images_path
        self.metadata_path = metadata_path
        self.transform = transform if transform is not None else v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.train_size = train_size
        self.train, self.test = self._train_test_split()

        self.train_dataset = CustomDataset(self.train[0], self.train[1], transform=self.transform)
        self.test_dataset = CustomDataset(self.test[0], self.test[1], transform=self.transform)

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