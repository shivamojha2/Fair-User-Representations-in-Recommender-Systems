from typing import List, Union

from loss import *
from torch import Tensor


class LossZoo:
    """
    LossZoo
    """

    loss_dict = {"ce_loss": CrossEntropyCriterion}

    def __new__(cls, loss: str, line_id_weights: Tensor, device: Union(List, str)):
        """
        Loss
        """
        assert loss in cls.loss_dict.keys(), "Invalid loss name"
        return cls.loss_dict[loss](line_id_weights)
