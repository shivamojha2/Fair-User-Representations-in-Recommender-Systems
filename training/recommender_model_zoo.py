"""
Recommender Model Zoo

@author: Shivam Ojha
"""
import torch
from recommender_model import MLP, PMF


class RecommenderModelZoo:
    """
    Model Zoo

    Returns:
        _type_: _description_
    """

    model_dict = {"PMF": PMF, "MLP": MLP}

    def __new__(cls, name, data_processor_dict, **config):
        assert name in cls.model_dict.keys(), "Invalid recommender model name"
        model = cls.model_dict[name](data_processor_dict, **config)
        # Initialize model params
        model.apply(model.init_weights)
        if torch.cuda.device_count() > 0:
            model = model.cuda()
        return model
