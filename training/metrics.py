"""
Metrics

author: Shivam Ojha
"""
from abc import abstractmethod
from collections import defaultdict

import torch


class Metric(object):
    """
    Metric class
    """

    @abstractmethod
    def compute(self, outputs, batch):
        """
        Returns dictionary of metrics to track

        Args:
            outputs (Tensor): raw logit outputs
            batch (Tensor): batch of ground truth labels
        """
        raise NotImplementedError

    def __call__(self, outputs, batch):
        return self.compute(outputs, batch)


class Metrics(Metric):
    """
    Child class

    Args:
        Metric (_type_): _description_
    """

    def __init__(self, thresh=0.5) -> None:
        super().__init__()
        self.thresh = thresh

    def compute(self, outputs, batch):

        # probs = outputs.logits.sigmoid()
        # labels = batch["labels"]

        # pred = torch.where(probs > self.thresh, 1, 0)

        metric_dict = defaultdict(list)
        # metric_dict["recall"]
        return metric_dict
