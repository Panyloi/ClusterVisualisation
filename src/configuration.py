import os

import yaml


class Singleton:

    def __new__(cls):
            if not hasattr(cls, 'instance'):
                cls.instance = super(Singleton, cls).__new__(cls)
            return cls.instance


class Configuration(Singleton):
    
    @classmethod
    def load(cls, path: str = os.path.dirname(__file__) + "/config.yaml"):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        cls.instance.__setattr__('config', dict(cfg))

    def __getitem__(self, key):
        return self.instance.config[key]
