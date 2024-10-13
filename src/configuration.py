"""
Module with functionalities for global configuration provider
"""

import os

import yaml

class Configuration:
    """
    Configuration class holds global fixed configuration
    """

    cfg : dict | None = None

    @classmethod
    def load(cls, path: str = os.path.dirname(__file__) + "/config.yaml"):
        """Load configuration from the specified YAML file."""
        with open(path, 'r') as f:
            cls.cfg = yaml.safe_load(f)

    @classmethod
    def get(cls, key):
        """Configuration values getter"""
        if cls.cfg is None:
            return None
        return cls.cfg[key]

    @classmethod
    def __class_getitem__(cls, key):
        """Configuration values subscript getter"""
        return cls.get(key)
