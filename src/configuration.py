import os

import yaml


class Configuration:
    
    cfg = None

    @classmethod
    def load(cls, path: str = os.path.dirname(__file__) + "/config.yaml"):
        """Load configuration from the specified YAML file."""
        with open(path, 'r') as f:
            cls.cfg = yaml.safe_load(f)
            
    @classmethod
    def __class_getitem__(cls, key):
        if cls.cfg is None:
            return None
        return cls.cfg[key]
