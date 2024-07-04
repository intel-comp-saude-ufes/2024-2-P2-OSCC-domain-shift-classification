from torch.nn import CrossEntropyLoss

class LossSelector:
    def __init__(self, loss_name, weight=None):
        self.loss_name = loss_name
        self.weight = weight

    def _select_loss(self):
        if self.loss_name == 'cross_entropy':
            return CrossEntropyLoss(weight=self.weight)
        else:
            raise ValueError('Invalid loss name')