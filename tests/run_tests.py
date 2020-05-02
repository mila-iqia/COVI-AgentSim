import os
import subprocess
from tempfile import TemporaryDirectory

# Force COLLECT_LOGS=True
# Force RISK_MODEL=True
# Fix RISK_MAPPING_FILE relative path
import covid19sim.config as config
config.COLLECT_LOGS = True
config.RISK_MODEL = "first order probabilistic tracing"
config.RISK_MAPPING_FILE = os.path.join(os.path.dirname(__file__), config.RISK_MAPPING_FILE)


def start_inference_server(root_dir=None):
    if root_dir is None:
        root_dir = os.path.dirname(__file__)
    working_dir = os.path.join(root_dir, "covid_p2p_risk_prediction")
    # TODO: use mila-iqia once satyaog's branch gets merged
    subprocess.run(
        f"git clone -binference-server --depth=1 "
        f"https://github.com/satyaog/covid_p2p_risk_prediction.git {working_dir}".split(),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Hack to have a config to use
    for cmd in ["git remote add ci_config "
                "https://github.com/satyaog/covid_p2p_risk_prediction.git",
                "git fetch --depth=1 ci_config ci_config"]:
        subprocess.run(cmd.split(), cwd=working_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        "git checkout ci_config/ci_config -- exp/DEBUG-0/Configurations/train_config.yml".split(),
        cwd=working_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return subprocess.Popen(
        "python server_bootstrap.py -w4 -p6688 -eexp/DEBUG-0 -v1".split(),
        cwd=working_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def test():
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='*_test.py')

    with TemporaryDirectory() as inference_server_d:
        server_process = start_inference_server(inference_server_d)
        assert server_process.returncode is None

        try:
            runner = unittest.TextTestRunner(verbosity=2)
            assert runner.run(suite).wasSuccessful()
        finally:
            server_process.kill()


if __name__ == "__main__":
    test()
