import os
import unittest

from covid19sim.configs.exp_config import ExpConfig


class Tests(unittest.TestCase):

    def setUp(self):
        if ExpConfig().is_empty():
            ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "../src/covid19sim/configs/test_config.yml"))

    def test_singleton(self):
        self.assertIs(ExpConfig(), ExpConfig())

    def test_getters_setters(self):
        _config = ExpConfig._instance._config
        instance = ExpConfig()
        
        for k, v in _config.items():
            self.assertEqual(ExpConfig.get(k), _config[k])
            self.assertEqual(instance[k], _config[k])


if __name__ == "__main__":
    unittest.main()
