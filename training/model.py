"""
Model    
"""
from typing import Callable

import torch.nn as nn


class Model(nn.Module):
    """
    Model
    """

    def __init__(self, config, loss_function: Callable) -> None:
        super().__init__()
        self.config = config
        self.loss_function = loss_function

    def predict(self):
        pass

    def forward(self):
        pass
