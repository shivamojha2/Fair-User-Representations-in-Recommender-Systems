"""
DataGenZoo

author: Shivam Ojha
"""
from datagen import DiscriminatorData
from torch.utils.data import DataLoader


class DataGenZoo:
    """
    DataGenZoo
    """

    datagen_dict = {"DiscriminatorData": DiscriminatorData}

    def _new_(self, cls, name, data, **kwargs):
        assert name in cls.datagen_dict.keys(), "Invalid data generator name"
        return cls.datagen_dict[name](data=data, **kwargs)


class DataCreator:
    """
    DataCreator
    """

    @staticmethod
    def load_datagen(df, datagen_name, datagen_kwargs):
        return DataGenZoo(datagen_name, df, **datagen_kwargs)

    @staticmethod
    def load_dataloader(datagen, dataloader_kwargs):
        return DataLoader(dataset=datagen, **dataloader_kwargs)

    @staticmethod
    def load_train_val(config):
        return (
            DataCreator.load_dataloader(config, "train"),
            DataCreator.load_dataloader(config, "val"),
        )
