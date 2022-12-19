"""
Torch utils
"""
import logging
import os
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter


def move_to_device(
    obj: Union[dict, list, torch.Tensor], device: torch.device
) -> Union[dict, list, torch.Tensor]:
    """
    Move objects to device

    Args:
        obj (Union[dict, list, torch.Tensor]): Object to move to device
        device (torch.device): Hardware device

    Returns:
        obj: Same obj as input with tensors moved to device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = move_to_device(v, device)
        return result
    elif isinstance(obj, list):
        result = []
        for v in obj:
            result.append(move_to_device(v, device))
        return result
    else:
        raise TypeError("Invalid type error")


def add_scalars(writer: SummaryWriter, iteration: int, prefix: str, **kwargs):
    """
    Add scalars to tensorboard

    Args:
        writer (SummaryWriter): Instance of pytorch SummaryWriter
        iteration (int): Current iteration
        prefix (str): Prefix for scalar group
    """
    for metric_name, value in kwargs.items():
        writer.add_scalar("{}/{}".format(prefix, metric_name), value, iteration)


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
