"""
Datagen
"""
from abc import abstractmethod

import pandas as pd
from torch.utils.data import Dataset


class BaseDatagen(Dataset):
    """
    Base Datagen

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, data):
        """
        Constructir for BaseDatagen
        """
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        return super().__getitem__(row)

    @abstractmethod
    def load_sample(self, row: pd.Series):
        """
        Abstract class to be overwritten by child class. Load input / target data from sample metadata

        Args:
            row (pd.Series): _description_
        """
        return

    @abstractmethod
    def preprocess_data(self, data):
        return data


class DiscriminatorData(BaseDatagen):
    def __init__(self, data):
        super().__init__(data)

    def preprocess_data(self, data):
        return super().preprocess_data(data)

    def load_sample(self, row: pd.Series):
        return super().load_sample(row)
    