import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyCriterion(nn.CrossEntropyLoss):
    """
    Cross Entropy Loss
    """

    def __init__(self, line_id_weights: Tensor):
        super().__init__()
        self.line_id_weights = line_id_weights

    def _compute_loss(self, prediction_scores, labels=None):
        ce_loss = None
        if labels is not None:
            loss_fct = F.cross_entropy
            prediction_score_flat = prediction_scores.flatten(0, 1)
            labels_flat = labels.view(-1)

            ce_loss = loss_fct(prediction_score_flat, labels_flat, self.line_id_weights)
        return ce_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self._compute_loss(input, labels=target)
