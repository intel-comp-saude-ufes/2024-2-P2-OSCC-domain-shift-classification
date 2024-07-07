import torch
from torch import nn
from torchvision.transforms import v2

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from src.data.dataset.custom_dataset import CustomDataset
from src.data.dataset.dataset_interface import DatasetInterface

class PatchDataset(DatasetInterface):
    name = 'patches_ndb'
    """
    Dataset for patches images. Arranges the patches in train and test sets considering the parent image (prefix of the image name). It has a train and test dataset that can be accessed by the attributes train_dataset and test_dataset.
    These datasets can be used in the DataLoader class from PyTorch.
    """	
    def __init__(self, path, train_size=0.8, k_folds=5, train_transform=None, test_transform=None):
        self.path = path
        self.k_folds = k_folds
        self.train_size = train_size
        
        self.train_transform = train_transform if train_transform is not None else v2.Compose([v2.ToTensor(),
                                                                                               v2.Resize((512,512)),
                                                                                               v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
        self.test_transform = test_transform if train_transform is not None else v2.Compose([v2.ToTensor(),
                                                                                             v2.Resize((512,512)),
                                                                                             v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.labels_names = {0: 'no_dysplasia', 1: 'carcinoma', 2: 'with_dysplasia'}
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
        carcinoma = list(self.path.glob('carcinoma/*.png'))

        # non-carcinoma images
        no_dysplasia = list(self.path.glob('no_dysplasia/*.png'))
        with_dysplasia = list(self.path.glob('with_dysplasia/*.png'))

        return carcinoma, no_dysplasia, with_dysplasia
    
    def _get_parent_image_name(self, image_name):
        """
        Get the parent image name from the image name
        """
        image_filename = image_name.parts[-1]
        image_root = "_".join(image_filename.split('_')[0:-1])
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

    def _create_label_from_path(self, image_class):
        """
        Create the labels for the dataset
        """
        
        name = str(image_class)
        if 'carcinoma' in name:
            return 1
        elif 'no_dysplasia' in name:
            return 0
        elif 'with_dysplasia' in name:
            return 2

    def _train_test_split(self):
        """
        Split the dataset into train and test
        """
        carcinoma, no_dysplasia, with_dysplasia = self._get_files()
        carcinoma_parent = list(set([self._get_parent_image_name(image) for image in carcinoma]))
        no_dysplasia_parent = list(set([self._get_parent_image_name(image) for image in no_dysplasia]))
        with_dysplasia_parent = list(set([self._get_parent_image_name(image) for image in with_dysplasia]))

        images = list(set(carcinoma_parent).union(no_dysplasia_parent).union(with_dysplasia_parent))
        
        # create arrays of 1 and 0 for carcinoma and non-carcinoma
        carinoma_parents_len = len(set(carcinoma_parent))
        no_dysplasia_parents_len = len(set(no_dysplasia_parent).difference(carcinoma_parent))
        with_dysplasia_parents_len = len(set(with_dysplasia_parent).difference(carcinoma_parent).difference(no_dysplasia_parent))

        carcinoma_labels = list(np.ones(carinoma_parents_len))
        no_dysplasia_labels = list(np.zeros(no_dysplasia_parents_len))
        with_dysplasia_labels = list(np.ones(with_dysplasia_parents_len) * 2)

        labels = carcinoma_labels + no_dysplasia_labels + with_dysplasia_labels

        # split the dataset by parent name
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, train_size=self.train_size, stratify=labels, random_state=42)
        
        # get patches images
        images_patches = carcinoma + no_dysplasia + with_dysplasia
        
        images_train = [image for image in images_patches if self._get_parent_image_name(image) in images_train]
        # 0, 1, 2 for no_dysplasia, carcinoma and with_dysplasia
        labels_train = [self._create_label_from_path(image) for image in images_train]

        images_test = [image for image in images_patches if self._get_parent_image_name(image) in images_test]
        labels_test = [self._create_label_from_path(image) for image in images_train]

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
        
        images = self.train[0]
        images_parents = list(set([self._get_parent_image_name(image) for image in images]))

        folds = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        folds_parents_list = []

        for fold in folds.split(images_parents):
            train_index, test_index = fold
            train_parents = [images_parents[i] for i in train_index]
            test_parents = [images_parents[i] for i in test_index]

            train_imgs = [image for image in images if self._get_parent_image_name(image) in train_parents]
            test_imgs = [image for image in images if self._get_parent_image_name(image) in test_parents]

            train_labels = [self._create_label_from_path(image) for image in train_imgs]
            test_labels = [self._create_label_from_path(image) for image in test_imgs]

            folds_parents_list.append((train_imgs, train_labels, test_imgs, test_labels))

        self._create_df_splits(folds_parents_list)

        return folds_parents_list
    
    def get_k_fold_train_val_tuple(self, k):
        """
        Get K-fold train and validation dataset
        """
        train_dataset = CustomDataset(self.k_folds_dataset[k][0], self.k_folds_dataset[k][1], transform=self.train_transform)
        val_dataset = CustomDataset(self.k_folds_dataset[k][2], self.k_folds_dataset[k][3], transform=self.test_transform)

        return train_dataset, val_dataset

    def __len__(self):
        return len(self.train[0]) + len(self.test[0])