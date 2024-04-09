import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class TrivalOptimizer(Optimizer):
    """
    An optimizer that do nothing
    """
    def __init__(self, params, lr=0.0):
        defaults = dict(lr=lr)
        super(TrivalOptimizer, self).__init__(params, defaults)
        
    def step(self, closure=None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def __repr__(self):
        return "TrivalOptimizer"
    
class TrivialScheduler(LambdaLR):
    """
    A scheduler that do nothing
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(TrivialScheduler, self).__init__(optimizer, lambda x: 1, last_epoch)
    
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def __repr__(self):
        return "TrivialScheduler"