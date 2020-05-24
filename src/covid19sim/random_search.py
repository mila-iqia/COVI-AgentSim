import os
import subprocess
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from covid19sim.utils import parse_search_configuration


def sample_param(sample_dict):
    """sample a value (hyperparameter) from the instruction in the
    sample dict:
    {
        "sample": "range | list",
        "from": [min, max, step] | [v0, v1, v2 etc.]
    }
    if range, as np.arange is used, "from" MUST be a list, but may contain
    only 1 (=min) or 2 (min and max) values, not necessarily 3

    Args:
        sample_dict (dict): instructions to sample a value

    Returns:
        scalar: sampled value
    """
    if not isinstance(sample_dict, dict) or "sample" not in sample_dict:
        return sample_dict
    if sample_dict["sample"] == "range":
        value = np.random.choice(np.arange(*sample_dict["from"]))
    elif sample_dict["sample"] == "list":
        value = np.random.choice(sample_dict["from"])
    elif sample_dict["sample"] == "uniform":
        value = np.random.uniform(*sample_dict["from"])
    else:
        raise ValueError("Unknonw sample type in dict " + str(sample_dict))
    return value


def load_search_conf(path):
    path = Path(path)
    assert path.exists()
    assert path.suffix in {".yaml", ".yml"}
    with path.open("r") as f:
        return yaml.safe_load(f)


def sample_search_conf(exp):
    conf = {}
    for k, v in exp.items():
        conf[k] = sample_param(v)
    return conf


@hydra.main(config_path="hydra-configs/search/config.yaml", strict=False)
def main(conf: DictConfig) -> None:

    """
                HOW TO USE

    $ python random_search.py exp_file=experiment n_search=20

    add `dev=True` to just see the commands that would be run, without
    running them

    NOTE: ALL parameters used in run.py may be overridden from this commandline.
    For instance you can change init_percent_sick

    $ python random_search.py exp_file=experiment n_search=20 init_percent_sick=0.1
    """

    # These will be filtered out when passing arguments to run.py
    RANDOM_SEARCH_SPECIFIC_PARAMS = {"n_search", "dev", "exp_file"}

    # move back to original directory because hydra moved
    os.chdir(hydra.utils.get_original_cwd())

    # get command-line arguments as native dict
    overrides = parse_search_configuration(conf)

    # load experimental configuration
    # override with exp_file=<X>
    # where <X> is in hydra-configs/search and is ".yaml"
    exp = load_search_conf(
        Path()
        / "hydra-configs"
        / "search"
        / (overrides.get("exp_file", "experiment") + ".yaml")
    )
    # override experimental parametrization with the commandline args
    exp.update(overrides)

    # run n_search jobs
    print("-" * 80)
    print("-" * 80)
    for i in range(conf.get("n_search", 1)):
        print("\nJOB", i)
        opts = sample_search_conf(exp)
        command = "sbatch job.sh"
        for k, v in opts.items():
            if k not in RANDOM_SEARCH_SPECIFIC_PARAMS:
                command += f" {k}={v}"
        if "dev" in exp and exp["dev"]:
            print(">>> ", command)
        else:
            process = subprocess.call(command.split())
        print()
        print("-" * 80)
        print("-" * 80)



if __name__ == "__main__":
    main()
