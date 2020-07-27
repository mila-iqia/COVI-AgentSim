import typing
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from covid19sim.utils.utils import parse_configuration

HYDRA_SIM_PATH = (
    Path(__file__).parent.parent.parent / "src/covid19sim/configs/simulation"
).resolve()


def get_test_conf(conf_name):
    """
    Loads the default configurations in configs and overwrites it
    with values in test_configs/`conf_name`

    conf_name **must** be in `tests/test_configs/`

    Args:
        conf_name (str): name of the configuration to load in `tests/test_configs/`

    Returns:
        dict: full overwritten config, using covid19sim.utils.parse_configuration
    """
    config_path = Path(__file__).parent.parent / "test_configs" / conf_name
    config_path = config_path.resolve()

    assert config_path.suffix == ".yaml"
    assert config_path.exists()

    config = HYDRA_SIM_PATH / "config.yaml"

    with config.open("r") as f:
        defaults = yaml.safe_load(f)["defaults"]

    default_confs = [
        OmegaConf.load(str(HYDRA_SIM_PATH / (d + ".yaml")))
        for d in defaults
    ]
    conf = OmegaConf.merge(*default_confs, OmegaConf.load(str(config_path)))

    return parse_configuration(conf)
