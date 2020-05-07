import os
from multiprocessing import Process

import covid19sim.config as config
import covid19sim.server_bootstrap

# Force test config value
config.COLLECT_LOGS = True
config.USE_INFERENCE_SERVER = True
config.COLLECT_TRAINING_DATA = True
config.INTERVENTION_DAY = 10
config.RISK_MODEL = "transformer"
config.UPDATES_PER_DAY = 1


def start_inference_server():
    exp_dir = os.path.join(os.path.dirname(__file__), "exp/DEBUG-0")
    p = Process(target=covid19sim.server_bootstrap.main, args=([f"-e{exp_dir}"],), daemon=True)
    p.start()
    return p


def test():
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
