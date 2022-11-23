"""
Datagen
"""
import pandas as pd
from base_datagen import BaseDatagen


class DiscriminatorData(BaseDatagen):
    def __init__(self, data):
        super().__init__(data)

    def preprocess_data(self, data):
        return super().preprocess_data(data)

    def load_sample(self, row: pd.Series):
        return super().load_sample(row)
    