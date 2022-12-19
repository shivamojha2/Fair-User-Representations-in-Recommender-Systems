"""
ModelZoo

@author: Shivam Ojha
"""
from typing import Callable

from model import Model
from transformers import (
    AdamW,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


class ModelZoo:
    """
    Model Zoo

    Returns:
        _type_: _description_
    """

    model_dict = {"Model": Model}

    def __new__(cls, name, model_kwargs, loss_function: Callable):
        assert name in cls.model_dict.keys(), "Invalid architecture name"
        config = PretrainedConfig(**model_kwargs)
        return cls.model_dict[name](config, loss_function)


class OptimizerZoo:
    """
    Optimizer Zoo
    """

    optimizer_dict = {"AdamW": AdamW}

    def __new__(cls, name, model_parameters, **optimizer_kwargs):
        assert name in cls.optimizer_dict.keys(), "Invalid optimizer name"
        return cls.optimizer_dict[name](model_parameters, **optimizer_kwargs)


class SchedulerZoo:
    """
    Scheduler Zoo
    """

    scheduler_dict = {
        "Linear": get_linear_schedule_with_warmup,
        "polynomial": get_polynomial_decay_schedule_with_warmup,
    }

    def __new__(
        cls, name, optimizer, num_warmup_steps, num_training_steps, **scheduler_kwargs
    ):
        assert name in cls.scheduler_dict.keys(), "Invalid scheduler name"
        return cls.scheduler_dict[name](
            optimizer, num_warmup_steps, num_training_steps, **scheduler_kwargs
        )
