"""
Recommender Model Zoo
"""
import torch
from recommender_model import MLP, PMF


class RecommenderModelZoo:
    """
    Recommender Model Zoo
    """

    model_dict = {"PMF": PMF, "MLP": MLP}

    def __new__(cls, name, data_processor_dict, exp_out_dir, **config):
        assert name in cls.model_dict.keys(), "Invalid recommender model name"
        model = cls.model_dict[name](data_processor_dict, exp_out_dir, **config)
        # Initialize model params
        model.apply(model.init_weights)
        if torch.cuda.device_count() > 0:
            model = model.cuda()
        return model
