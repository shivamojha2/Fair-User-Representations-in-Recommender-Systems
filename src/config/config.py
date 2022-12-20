"""
Config
"""
import json


class Config:
    """
    Class to abstract away methods in config file
    """

    def __init__(self, json_path):
        self.json_path = json_path
        json_file = self.load_json(json_path)
        for key, value in json_file.items():
            setattr(self, key, value)

    def load_json(self, json_path):
        """
        Load json file from given json_path

        Args:
            json_path (str): Path to json file

        Returns:
            dict: json_file dict
        """
        return json.load(open(json_path, "r"))
