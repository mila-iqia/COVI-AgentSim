# Force COLLECT_LOGS=True
# Force RISK_MODEL=True
import config
config.COLLECT_LOGS = True
config.RISK_MODEL = "first order probabilistic tracing"


def test():
    import unittest
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='*_test.py')

    runner = unittest.TextTestRunner()
    assert runner.run(suite).wasSuccessful()


if __name__ == "__main__":
    test()
