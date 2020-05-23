import subprocess
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from covid19sim.utils import parse_search_configuration

@hydra.main(config_path="hydra-configs/search", config_name="config")
def main(conf: DictConfig) -> int:

    os.chdir(hydra.utils.get_original_cwd())
    conf = parse_search_configuration(conf)

    command = "python run.py"
    for k, v in conf.items():
        command += f" {k}={v}"

    process = subprocess.call(command.split())

    print()
    print()
    print()
    return 0  # required by sweeper


if __name__ == "__main__":
    main()
