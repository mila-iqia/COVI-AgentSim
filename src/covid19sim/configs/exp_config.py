"""
[summary]
"""

import yaml


class ExpConfig(object):
    """
    [summary]
    """
    config = None

    @classmethod
    def __getitem__(cls, key):
        """
        [summary]

        Args:
            key ([type]): [description]

        Returns:
            [type]: [description]
        """
        return cls.config[key]

    @classmethod
    def __setitem__(cls, key, item):
        """
        [summary]

        Args:
            key ([type]): [description]
            item ([type]): [description]
        """
        cls.config[key] = item

    @classmethod
    def get(cls, key):
        """
        [summary]

        Args:
            key ([type]): [description]

        Returns:
            [type]: [description]
        """
        return cls.config[key]

    @classmethod
    def set(cls, key, val):
        """
        [summary]

        Args:
            key ([type]): [description]
            val ([type]): [description]
        """
        cls.config[key] = val

    @classmethod
    def load_config(cls, path):
        """
        [summary]

        Args:
            path ([type]): [description]
        """
        with open(path) as file:
            ExpConfig.config = yaml.load(file, Loader=yaml.FullLoader)
        return ExpConfig.config
