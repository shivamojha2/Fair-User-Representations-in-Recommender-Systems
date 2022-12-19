"""
Torch utils
"""
import logging
import os
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter


def move_batch_to_gpu(batch):
    """
    Move batch to gpu device
    """
    if torch.cuda.device_count() > 0:
        for c in batch:
            if type(batch[c]) is torch.Tensor:
                batch[c] = batch[c].cuda()
    return batch


def save_model(epoch, model, optimizer, scheduler, save_dir):
    """
    Save model

    Args:
        epoch (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_
        save_dir (_type_): _description_
    """
    model_parent_dir = os.path.join(save_dir, "models")
    if not os.path.exists(model_parent_dir):
        os.makedirs(model_parent_dir)
    # Save Model
    logging.info("Saving model for epoch %s", epoch)
    states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimzer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(states, os.path.join(model_parent_dir, f"model_{epoch}.pth"))
