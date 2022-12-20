"""
Datagen Zoo
"""
from datagen import DiscriminatorDataset, MovieLens1MDataset


class DataGenZoo:
    """
    DataGenZoo
    """

    datagen_dict = {
        "MovieLens1MDataset": MovieLens1MDataset,
        "DiscriminatorDataset": DiscriminatorDataset,
    }

    def __new__(cls, name, path, dataset, stage, batch_size, num_neg):
        assert name in cls.datagen_dict.keys(), "Invalid data generator name"
        return cls.datagen_dict[name](path, dataset, stage, batch_size, num_neg)
