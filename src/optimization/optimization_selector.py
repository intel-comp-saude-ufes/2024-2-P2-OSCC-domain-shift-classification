from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class OptimizationSelector:
    def __init__(self, optimizer_name, scheduler_name, learning_rate, weight_decay, **kwargs):
        self.optimization_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kwargs = kwargs

        self.optimizer = None
        self.scheduler = None

    def get_optimizer(self, model_parameters):
        if self.optimization_name == 'adam':
            self.optimizer = Adam(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimization_name == 'sgd':
            self.optimizer = SGD(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError('Invalid optimization name')
        
        return self.optimizer

    def get_scheduler(self, optimizer):
        if not self.optimizer:
            raise ValueError('Optimizer is not initialized')
        
        if self.step_size and self.gamma:
            step_size = self.kwargs.step_size.get('step_size', 30)
            gamma = self.kwargs.gamma.get('gamma', 0.1)
            self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            mode = self.kwargs.mode.get('mode', 'min')
            factor = self.kwargs.factor.get('factor', 0.1)
            patience = self.kwargs.patience.get('patience', 10)
            verbose = self.kwargs.verbose.get('verbose', True)
            self.scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)

        return self.scheduler