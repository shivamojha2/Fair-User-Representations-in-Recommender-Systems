"""
MetricsZoo

author: Shivam Ojha
"""

from metrics import *


class ModelMetricsZoo:
    """
    DataGenZoo
    """

    metric_dict = {"Metrics": Metrics}

    def __new__(cls, name):
        assert name in cls.metric_dict.keys(), "Invalid metric name"
        return cls.metric_dict[name]()
