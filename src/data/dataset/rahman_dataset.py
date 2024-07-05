import torch
from torchvision.transforms import v2

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from src.data.dataset.custom_dataset import CustomDataset
from src.data.dataset.dataset_interface import DatasetInterface

class RahmanDataset(DatasetInterface):
    name = 'rahman'
    """
    Dataset for patches images. Arranges the patches in train and test sets considering the parent image (prefix of the image name). It has a train and test dataset that can be accessed by the attributes train_dataset and test_dataset.
    These datasets can be used in the DataLoader class from PyTorch.
    """	

    def __init__(self, path, train_size=0.8, k_folds=5, train_transform=None, test_transform=None):
        self.path = path
        self.k_folds = k_folds
        self.train_size = train_size
        self.train_transform = train_transform if train_transform is not None else v2.Compose([v2.ToImage(),v2.Resize((512,512))])
        self.test_transform = test_transform if train_transform is not None else v2.Compose([v2.ToImage(),v2.Resize((512,512))])
        self.labels_names = {0: 'noncarcinoma', 1: 'carcinoma'}
        self.train, self.test = self._train_test_split()

        self.train_dataset = CustomDataset(self.train[0], self.train[1], transform=self.train_transform)
        self.test_dataset = CustomDataset(self.test[0], self.test[1], transform=self.test_transform)

        self.folds_df = None
        self.k_folds_dataset = self._generate_k_folds()

    
    def _get_files(self):
        """
        Get files from the path by class carcinoma and non-carcinoma
        """
        # carcinoma images
        carcinoma = list(self.path.glob("First Set/100x OSCC Histopathological Images/*.jpg"))
        carcinoma.extend(list(self.path.glob('Second Set/400x OSCC Histopathological Images/*.jpg')))
        
        # non-carcinoma images
        noncarcinoma = list(self.path.glob('First Set/100x Normal Oral Cavity Histopathological Images/*.jpg'))
        noncarcinoma.extend(list(self.path.glob('Second Set/400x Normal Oral Cavity Histopathological Images/*.jpg')))

        return carcinoma, noncarcinoma

    def _train_test_split(self):
        """
        Split the dataset into train and test
        """
        # get file complete path for each image file 
        carcinoma, noncarcinoma = self._get_files()

        # create arrays of 1 and 0 for carcinoma and non-carcinoma
        carcinoma_labels = list(np.ones(len(carcinoma)))
        noncarcinoma_labels = list(np.zeros(len(noncarcinoma)))

        # extend the labels
        labels = carcinoma_labels + noncarcinoma_labels
        images = carcinoma + noncarcinoma

        # split the dataset by parent name
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, train_size=self.train_size, stratify=labels, random_state=42)

        return (images_train, labels_train), (images_test, labels_test)

    def _create_df_splits(self, k_folds_datasets):
        """
        Create dataframe with the k-folds splits for the dataset
        """
        folds = [(i, fold) for i, fold in enumerate(k_folds_datasets)]
        folds = [(fold[0], fold[1][2], fold[1][3]) for fold in folds]

        # expand the folds
        folds = [(fold[0], image.parts[-1], label) for fold in folds for image, label in zip(fold[1], fold[2])]

        self.folds_df = pd.DataFrame(folds, columns=['fold', 'image_path', 'class'])

    def _generate_k_folds(self):
        """
        Generate k-folds for the dataset
        """
        
        images, labels = self.train
        images = self.train[0]

        folds = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        folds_list = []

        for j, fold in enumerate(folds.split(images, labels)):
            train_index, test_index = fold

            train_imgs = [images[i] for i in train_index]
            test_imgs = [images[i] for i in test_index]

            train_labels = [labels[i] for i in train_index]
            test_labels = [labels[i] for i in test_index]

            folds_list.append((train_imgs, train_labels, test_imgs, test_labels))

        self._create_df_splits(folds_list)

        return folds_list
    
    def get_k_fold_train_val_tuple(self, k):
        """
        Get K-fold train and validation dataset
        """
        train_dataset = CustomDataset(self.k_folds_dataset[k][0], self.k_folds_dataset[k][1], transform=self.train_transform)
        val_dataset = CustomDataset(self.k_folds_dataset[k][2], self.k_folds_dataset[k][3], transform=self.test_transform)

        return train_dataset, val_dataset


    def __len__(self):
        return len(self.train[0]) + len(self.test[0])