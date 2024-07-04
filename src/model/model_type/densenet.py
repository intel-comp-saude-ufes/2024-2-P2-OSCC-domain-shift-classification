from torchvision import models
from torch import nn
import torch

import os

class DenseNet121:
    name = 'densenet121'
    """
    DenseNet121 model from torchvision, with the classifier layer changed to the number of classes in the dataset and the weights loaded from the default weights or a custom path.
    """
    def __init__(self, num_classes=2, weights='default', device='auto') -> None:
        self.model = self._load_model(weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.features = self.model.features
        self.classifier = self.model.classifier

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if torch.cuda.device_count() > 1:
            self.model= torch.nn.DataParallel(self.model).to(self.device)
        else:
            self.model.to(self.device)
    
    def _load_model(self, weights):
        """
        Load the DenseNet121 model from torchvision with the weights from the default or a custom path.
        """
        if weights == 'default':
            return models.densenet121(weights='IMAGENET1K_V1')
        else:
            if not os.path.exists(weights):
                raise FileNotFoundError(f"File {weights} not found")
            model = models.densenet121()
            model.load_state_dict(torch.load(weights))
            return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_model(self):
        return self.model
    
    def parameters(self):
        return self.model.parameters()