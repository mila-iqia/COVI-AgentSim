import os
import subprocess
from pathlib import Path
import tempfile
import hydra
import numpy as np
import yaml
from omegaconf import DictConfig
import datetime

from covid19sim.utils import parse_search_configuration


def now_str():
    now = str(datetime.datetime.now())
    now = now.split(".")[0]
    now = now.replace("-", "").replace(":", "").replace(" ", "_")
    return now


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
    """
    Load a yaml file in `path`

    Args:
        path (str | pathlib.Path): path to yaml file

    Returns:
        any: python native variable loaded by PyYaml
    """
    path = Path(path)
    assert path.exists()
    assert path.suffix in {".yaml", ".yml"}
    with path.open("r") as f:
        return yaml.safe_load(f)


def sample_search_conf(exp):
    """
    Samples parameters parametrized in `exp`: should be a dict with
    values which fit `sample_params(dic)`'s API

    Args:
        exp (dict): experiment's parametrization

    Returns:
        dict: sampled configuration
    """
    conf = {}
    for k, v in exp.items():
        conf[k] = sample_param(v)
    return conf


def load_template(infra):
    """
    Get the template string to format according to arguments

    Returns:
        str: template string full of "{variable_name}"
    """
    if infra == "mila":
        with (Path(__file__).parent / "job_scripts" / "mila_sbatch_template.sh").open(
            "r"
        ) as f:
            return f.read()
    if infra == "intel":
        with (Path(__file__).parent / "job_scripts" / "intel_template.sh").open(
            "r"
        ) as f:
            return f.read()
    raise ValueError("Unknown infrastructure " + str(infra))


def fill_intel_template(template_str, conf):
    home = os.environ.get("HOME")

    env_name = conf.get("env_name", "covid")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))
    weights = conf.get("weights", str(Path(home) / "FRESH-SNOWFLAKE-224B/"))
    server_out = str(
        Path(home) / "covi_search" / "logs" / f"server_{conf['now_str']}.out"
    )

    if "dev" in conf and conf["dev"]:
        print(
            "Using:\n"
            + "\n".join(
                [
                    "  {:10}: {}".format("env_name", env_name),
                    "  {:10}: {}".format("code_loc", code_loc),
                    "  {:10}: {}".format("weights", weights),
                ]
            )
        )

    return template_str.format(env_name=env_name, code_loc=code_loc, weights=weights)


def fill_mila_template(template_str, conf):
    """
    Formats the template_str with variables from the conf dict,
    which is a sampled experiment

    Args:
        template_str (str): sbatch template
        conf (dict): sbatch parameters

    Returns:
        str: formated template
    """
    user = os.environ.get("USER")
    home = os.environ.get("HOME")

    partition = conf.get("partition", "main")
    cpu = conf.get("cpus", 6)
    mem = conf.get("mem", 16)
    gres = conf.get("gres", "")
    time = conf.get("time", "4:00:00")
    slurm_log = conf.get("slurm_log", f"/network/tmp1/{user}/covi-slurm-%j.out")
    env_name = conf.get("env_name", "covid")
    weights = conf.get("weights", f"/network/tmp1/{user}/FRESH-SNOWFLAKE-224B")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))

    if "dev" in conf and conf["dev"]:
        print(
            "Using:\n"
            + "\n".join(
                [
                    "  {:10}: {}".format("partition", partition),
                    "  {:10}: {}".format("cpus-per-task", cpu),
                    "  {:10}: {}".format("mem", mem),
                    "  {:10}: {}".format("gres", gres),
                    "  {:10}: {}".format("time", time),
                    "  {:10}: {}".format("slurm_log", slurm_log),
                    "  {:10}: {}".format("env_name", env_name),
                    "  {:10}: {}".format("code_loc", code_loc),
                    "  {:10}: {}".format("weights", weights),
                ]
            )
        )

    partition = f"#SBATCH --partition={partition}"
    cpu = f"#SBATCH --cpus-per-task={cpu}"
    mem = f"#SBATCH --mem={mem}GB"
    gres = f"#SBATCH --gres={gres}" if gres else ""
    time = f"#SBATCH --time={time}"
    slurm_log = f"#SBATCH -o {slurm_log}\n#SBATCH -e {slurm_log}"
    return template_str.format(
        partition=partition,
        cpu=cpu,
        mem=mem,
        gres=gres,
        time=time,
        slurm_log=slurm_log,
        env_name=env_name,
        code_loc=code_loc,
        weights=weights,
    )


def get_hydra_args(opts, exclude=set()):
    hydra_args = ""
    for k, v in opts.items():
        if k not in exclude:
            hydra_args += f" {k}={v}"
    return hydra_args


def printlines():
    print("=" * 80)
    print("=" * 80)


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

    NOTE: you may also pass arguments overridding the default `sbatch` job's
    parametrization like partition, gres, code_loc (=where is the simulator's code),
    env_name (= what conda env to load). For instance:

    $ python random_search.py partition=unkillable gres=gpu:1 env_name=covid-env\
                              n_search=20 init_percent_sick=0.1

    """

    # These will be filtered out when passing arguments to run.py
    RANDOM_SEARCH_SPECIFIC_PARAMS = {
        "n_search",  # number of random iterations
        "dev",  # dev-mode: print stuff, don't run them
        "exp_file",  # what experimental parametrization
        "partition",  # sbatch partition to use
        "cpus",  # sbatch number of cpus
        "mem",  # sbatch memory to request
        "time",  # sbatch job upper bound on duration
        "slurm_log",  # sbatch logs destination
        "gres",  # sbatch gres arg, may be nothing or gpu:1
        "env_name",  # conda environment to load
        "code_loc",  # where to find the source code, will cd there
        "weights",  # where to find the transformer's weights. default is /network/tmp1/<user>/FRESH-SNOWFLAKE-224B
        "infra",  # using Mila or Intel cluster?
    }

    # move back to original directory because hydra moved
    os.chdir(hydra.utils.get_original_cwd())

    # get command-line arguments as native dict
    overrides = parse_search_configuration(conf)

    # load experimental configuration
    # override with exp_file=<X>
    # where <X> is in hydra-configs/search and is ".yaml"
    conf = load_search_conf(
        Path()
        / "hydra-configs"
        / "search"
        / (overrides.get("exp_file", "experiment") + ".yaml")
    )
    # override experimental parametrization with the commandline conf
    conf.update(overrides)
    infra = conf.get("infra", "mila")
    template_str = load_template(infra)

    home = os.environ["HOME"]

    # run n_search jobs
    printlines()
    intel_str = ""
    for i in range(conf.get("n_search", 1)):

        # sample parameters
        opts = sample_search_conf(conf)
        # fill-in template with `partition` `time` `code_loc` etc. from command-line overwrites
        # get temporary file to write sbatch run file
        hydra_args = get_hydra_args(opts, RANDOM_SEARCH_SPECIFIC_PARAMS)
        if infra == "mila":
            print("\nJOB", i)
            job_str = fill_mila_template(template_str, conf)
            tmp = Path(tempfile.NamedTemporaryFile(suffix=".sh").name)

            # create temporary sbatch file
            with tmp.open("w") as f:
                f.write(job_str)

            command = f"sbatch {str(tmp)} {hydra_args}"

            # dev-mode: don't actually run the command
            if "dev" in conf and conf["dev"]:
                print("\n>>> ", command, end="\n\n")
                print(str(tmp))
                print("." * 50)
                print(job_str)
                print("." * 50)
            else:
                # not dev-mode: sbatch it!
                process = subprocess.call(command.split(), cwd=home)

            # prints
            print()
            printlines()

        if infra == "intel":
            if i == 0:
                conf["now_str"] = now_str()
                intel_str = fill_intel_template(template_str, opts)
            intel_str += "\n{}\n".format("python run.py " + hydra_args)

    if infra == "intel":
        path = Path(home) / "covi_search" / f"covi_search{conf['now_str']}.sh"
        assert not path.exists()
        if "dev" in conf and conf["dev"]:
            print(str(path))
            print("." * 50)
            print(intel_str)
            print("." * 50)

        else:
            command = f"sh {path.name}"
            with path.open("w") as f:
                f.write(intel_str)
            process = subprocess.call(command.split(), cwd=str(path.parent))


if __name__ == "__main__":
    main()
