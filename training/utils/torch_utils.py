"""
Torch utils
"""
import torch


def move_batch_to_gpu(batch):
    """
    Move batch to gpu device
    """
    if torch.cuda.device_count() > 0:
        for _b in batch:
            if type(batch[_b]) is torch.Tensor:
                batch[_b] = batch[_b].cuda()
    return batch
