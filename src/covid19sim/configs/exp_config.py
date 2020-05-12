"""
[summary]
"""

import yaml


class ExpConfig(object):
    """
    [summary]
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._config = {}
        return cls._instance

    def __getitem__(self, key):
        """
        [summary]

        Args:
            key ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self._config[key]

    def __setitem__(self, key, item):
        """
        [summary]

        Args:
            key ([type]): [description]
            item ([type]): [description]
        """
        self._config[key] = item

    def is_empty(self):
        return not len(self._config)

    @classmethod
    def get(cls, key):
        """
        [summary]

        Args:
            key ([type]): [description]

        Returns:
            [type]: [description]
        """
        return ExpConfig()[key]

    @classmethod
    def set(cls, key, val):
        """
        [summary]

        Args:
            key ([type]): [description]
            val ([type]): [description]
        """
        ExpConfig()[key] = val

    @classmethod
    def load_config(cls, path):
        """
        [summary]

        Args:
            path ([type]): [description]
        """
        config = ExpConfig()
        with open(path) as file:
            config._config = yaml.load(file, Loader=yaml.FullLoader)
