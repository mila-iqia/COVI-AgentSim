import yaml


class ExpConfig(object):
    config = None

    @classmethod
    def __getitem__(cls, key):
        return cls.config[key]

    @classmethod
    def __setitem__(cls, key, item):
        cls.config[key] = item

    @classmethod
    def get(cls, key):
        return cls.config[key]

    @classmethod
    def set(cls, key, val):
        cls.config[key] = val

    @classmethod
    def load_config(cls, path):
        with open(path) as file:
            ExpConfig.config = yaml.load(file, Loader=yaml.FullLoader)
