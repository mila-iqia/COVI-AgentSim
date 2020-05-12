"""
[summary]
"""
import os
from multiprocessing import Process

import covid19sim.server_bootstrap
from covid19sim.configs.exp_config import ExpConfig

# Load the experimental configuration
ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "src/covid19sim/configs/test_config.yml"))


def start_inference_server():
    """
    [summary]

    Returns:
        [type]: [description]
    """
    exp_dir = os.path.join(os.path.dirname(__file__), "exp/DEBUG-0")
    p = Process(target=covid19sim.server_bootstrap.main, args=([f"-e{exp_dir}"],), daemon=True)
    p.start()
    return p


def test():
    """
    [summary]
    """
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='*_test.py')

    server_process = start_inference_server()
    assert server_process.is_alive()

    try:
        runner = unittest.TextTestRunner(verbosity=2)
        assert runner.run(suite).wasSuccessful()
    finally:
        server_process.kill()


if __name__ == "__main__":
    test()
