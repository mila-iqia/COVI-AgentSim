"""
[summary]
"""
import os
from multiprocessing import Process

import covid19sim.server_bootstrap
from covid19sim.configs.exp_config import ExpConfig

# Load the experimental configuration
ExpConfig = ExpConfig()
ExpConfig.load_config(os.path.join(os.path.dirname(__file__), "src/covid19sim/configs/test_config.yml"))

# Fix relative paths
ExpConfig['CLUSTER_PATH'] = os.path.join(os.path.dirname(__file__), ExpConfig['CLUSTER_PATH'])
ExpConfig['TRANSFORMER_EXP_PATH'] = os.path.join(os.path.dirname(__file__), ExpConfig['TRANSFORMER_EXP_PATH'])


def start_inference_server():
    """
    Starts the inference server

    Returns:
        multiprocessing.Process: the process holding the inference server
    """
    exp_dir = ExpConfig['TRANSFORMER_EXP_PATH']
    p = Process(target=covid19sim.server_bootstrap.main, args=([f"-e{exp_dir}"],), daemon=True)
    p.start()
    return p


def test():
    """
    Run all *_test.py files from the tests/ directory
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
