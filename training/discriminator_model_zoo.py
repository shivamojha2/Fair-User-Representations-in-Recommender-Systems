"""
ModelZoo
"""
import torch
from discriminator_model import (
    MovieLens1MBinaryAttacker,
    MovieLens1MDiscriminatorModel,
    MovieLens1MMultiClassAttacker,
)


class DiscriminatorModelZoo:
    """
    Model Zoo

    Returns:
        _type_: _description_
    """

    model_dict = {
        "MovieLens1MDiscriminatorModel": MovieLens1MDiscriminatorModel,
        "MovieLens1MBinaryAttacker": MovieLens1MBinaryAttacker,
        "MovieLens1MMultiClassAttacker": MovieLens1MMultiClassAttacker,
    }

    def __new__(cls, name, exp_out_dir, **config):
        assert name in cls.model_dict.keys(), "Invalid discriminator model name"
        model = cls.model_dict[name](exp_out_dir, **config)
        # initiliaze model params
        model.apply(model.init_weights)
        if torch.cuda.device_count() > 0:
            model = model.cuda()
        return model
