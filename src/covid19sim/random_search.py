import os
import subprocess
from pathlib import Path
import tempfile
import hydra
import numpy as np
import yaml
from omegaconf import DictConfig
import datetime
import itertools
from covid19sim.utils import parse_search_configuration
import time


def now_str():
    now = str(datetime.datetime.now())
    now = now.replace("-", "").replace(":", "").replace(" ", "_").replace(".", "_")
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

    if sample_dict["sample"] == "cartesian":
        assert isinstance(
            sample_dict["from"], list
        ), "{}'s `from` field MUST be a list, found {}".format(
            sample_dict["sample"], sample_dict["from"]
        )
        return "__cartesian__"

    if sample_dict["sample"] == "sequential":
        assert isinstance(
            sample_dict["from"], list
        ), "{}'s `from` field MUST be a list, found {}".format(
            sample_dict["sample"], sample_dict["from"]
        )
        return "__sequential__"

    if sample_dict["sample"] == "range":
        return np.random.choice(np.arange(*sample_dict["from"]))

    if sample_dict["sample"] == "list":
        return np.random.choice(sample_dict["from"])

    if sample_dict["sample"] == "uniform":
        return np.random.uniform(*sample_dict["from"])

    raise ValueError("Unknown sample type in dict " + str(sample_dict))


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


def sample_cartesians(cartesian_keys, exp, idx):
    """
    Returns the `idx`th item in the cartesian product of all cartesian keys to
    be sampled.

    Args:
        cartesian_keys (list): keys in the experimental configuration that are to be used in the full
           cartesian product
        exp (dict): experimental configuration
        idx (int): index of the current sample

    Returns:
        dict: sampled point in the cartesian space (with keys = cartesian_keys)
    """
    conf = {}
    cartesian_values = [exp[key]["from"] for key in cartesian_keys]
    product = list(itertools.product(*cartesian_values))
    for k, v in zip(cartesian_keys, product[idx % len(product)]):
        conf[k] = v
    return conf


def sample_sequentials(sequential_keys, exp, idx):
    """
    Samples sequentially from the "from" values specified in each key of the experimental
    configuration which have sample == "sequential"
    Unlike `cartesian` sampling, `sequential` sampling iterates *independently* over each keys

    Args:
        sequential_keys (list): keys to be sampled sequentially
        exp (dict): experimental config
        idx (int): index of the current sample

    Returns:
        conf: sampled dict
    """
    conf = {}
    for k in sequential_keys:
        v = exp[k]["from"]
        conf[k] = v[idx % len(v)]
    return conf


def get_uuid():
    return "{}_{}".format(np.random.randint(1e5, 1e6), np.random.randint(1e5, 1e6))


def ipc_addresses():
    uuid = get_uuid()
    ipc_front = "ipc:///tmp/covid19_{}_frontend.ipc".format(uuid)
    ipc_back = "ipc:///tmp/covid19_{}_backend.ipc".format(uuid)
    return (ipc_front, ipc_back)


def sample_search_conf(exp, idx=0):
    """
    Samples parameters parametrized in `exp`: should be a dict with
    values which fit `sample_params(dic)`'s API

    Args:
        exp (dict): experiment's parametrization
        idx  (int): experiment's idx in the sampling procedure (useful in case a key
            should be sampled in a cartesian or sequential manner)

    Returns:
        dict: sampled configuration
    """
    conf = {}
    cartesians = []
    sequentials = []
    for k, v in exp.items():
        candidate = sample_param(v)
        if candidate == "__cartesian__":
            cartesians.append(k)
        elif candidate == "__sequential__":
            sequentials.append(k)
        else:
            conf[k] = candidate
    if sequentials:
        conf.update(sample_sequentials(sequentials, exp, idx))
    if cartesians:
        conf.update(sample_cartesians(cartesians, exp, idx))
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
    if infra == "beluga":
        with (Path(__file__).parent / "job_scripts" / "beluga_sbatch_template.sh").open(
            "r"
        ) as f:
            return f.read()
    raise ValueError("Unknown infrastructure " + str(infra))


def fill_intel_template(template_str, conf):
    home = os.environ.get("HOME")

    env_name = conf.get("env_name", "covid")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))
    weights = conf.get("weights", str(Path(home) / "FRESH-SNOWFLAKE-224B/"))
    ipc = conf.get("ipc", {"frontend": "", "backend": ""})
    cpu = conf.get("cpus", 6)
    use_transformer = str(conf.get("use_transformer", True)).lower()

    if conf.get("parallel_search"):
        workers = cpu - conf.get("n_runs_per_search", 1)
    else:
        workers = cpu - 1

    if "dev" in conf and conf["dev"]:
        print(
            "Using:\n"
            + "\n".join(
                [
                    "  {:10}: {}".format("env_name", env_name),
                    "  {:10}: {}".format("code_loc", code_loc),
                    "  {:10}: {}".format("weights", weights),
                    "  {:10}: {}".format("use_transformer", use_transformer),
                    "  {:10}: {}".format("workers", workers),
                    "  {:10}: {}".format("frontend", ipc["frontend"]),
                    "  {:10}: {}".format("backend", ipc["backend"]),
                ]
            )
        )
    frontend = '--frontend="{}"'.format(ipc["frontend"]) if ipc["frontend"] else ""
    backend = '--backend="{}"'.format(ipc["backend"]) if ipc["backend"] else ""

    return template_str.format(
        env_name=env_name,
        code_loc=code_loc,
        weights=weights,
        frontend=frontend,
        backend=backend,
        use_transformer=use_transformer,
        workers=workers,
    )


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
    time = str(conf.get("time", "4:00:00"))
    slurm_log = conf.get("slurm_log", f"/network/tmp1/{user}/covi-slurm-%j.out")
    if "%j.out" not in slurm_log:
        slurm_log = str(Path(slurm_log).resolve() / "covi-slurm-%j.out")
        if not Path(slurm_log).parent.exists():
            Path(slurm_log).parent.mkdir(parents=True)
    env_name = conf.get("env_name", "covid")
    weights = conf.get("weights", f"/network/tmp1/{user}/FRESH-SNOWFLAKE-224B")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))
    ipc = conf.get("ipc", {"frontend": "", "backend": ""})
    use_transformer = str(conf.get("use_transformer", True)).lower()
    workers = cpu - 1

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
                    "  {:10}: {}".format("frontend", ipc["frontend"]),
                    "  {:10}: {}".format("backend", ipc["backend"]),
                    "  {:10}: {}".format("use_transformer", use_transformer),
                    "  {:10}: {}".format("workers", workers),
                ]
            )
        )

    partition = (
        f"#SBATCH --partition={partition}"
        if partition != "covid"
        else "#SBATCH --reservation=covid\n#SBATCH --partition=long"
    )
    cpu = f"#SBATCH --cpus-per-task={cpu}"
    mem = f"#SBATCH --mem={mem}GB"
    gres = f"#SBATCH --gres={gres}" if gres else ""
    time = f"#SBATCH --time={time}"
    slurm_log = f"#SBATCH -o {slurm_log}\n#SBATCH -e {slurm_log}"
    frontend = '--frontend="{}"'.format(ipc["frontend"]) if ipc["frontend"] else ""
    backend = '--backend="{}"'.format(ipc["backend"]) if ipc["backend"] else ""
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
        frontend=frontend,
        backend=backend,
        use_transformer=use_transformer,
        workers=workers,
    )


def fill_beluga_template(template_str, conf):
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

    cpu = conf.get("cpus", 6)
    mem = conf.get("mem", 16)
    time = str(conf.get("time", "3:00:00"))
    slurm_log = conf.get("slurm_log", f"/scratch/{user}/covi-slurm-%j.out")
    if "%j.out" not in slurm_log:
        slurm_log = str(Path(slurm_log).resolve() / "covi-slurm-%j.out")
        if not Path(slurm_log).parent.exists():
            Path(slurm_log).parent.mkdir(parents=True)
    env_name = conf.get("env_name", "covid")
    weights = conf.get("weights", f"/scratch/{user}/FRESH-SNOWFLAKE-224B")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))
    ipc = conf.get("ipc", {"frontend": "", "backend": ""})
    use_transformer = str(conf.get("use_transformer", True)).lower()
    workers = cpu - 1

    if "dev" in conf and conf["dev"]:
        print(
            "Using:\n"
            + "\n".join(
                [
                    "  {:10}: {}".format("cpus-per-task", cpu),
                    "  {:10}: {}".format("mem", mem),
                    "  {:10}: {}".format("time", time),
                    "  {:10}: {}".format("slurm_log", slurm_log),
                    "  {:10}: {}".format("env_name", env_name),
                    "  {:10}: {}".format("code_loc", code_loc),
                    "  {:10}: {}".format("weights", weights),
                    "  {:10}: {}".format("frontend", ipc["frontend"]),
                    "  {:10}: {}".format("backend", ipc["backend"]),
                    "  {:10}: {}".format("use_transformer", use_transformer),
                    "  {:10}: {}".format("workers", workers),
                ]
            )
        )

    cpu = f"#SBATCH --cpus-per-task={cpu}"
    mem = f"#SBATCH --mem={mem}GB"
    time = f"#SBATCH --time={time}"
    slurm_log = f"#SBATCH -o {slurm_log}\n#SBATCH -e {slurm_log}"
    frontend = '--frontend="{}"'.format(ipc["frontend"]) if ipc["frontend"] else ""
    backend = '--backend="{}"'.format(ipc["backend"]) if ipc["backend"] else ""
    return template_str.format(
        cpu=cpu,
        mem=mem,
        time=time,
        slurm_log=slurm_log,
        env_name=env_name,
        code_loc=code_loc,
        weights=weights,
        frontend=frontend,
        backend=backend,
        use_transformer=use_transformer,
        workers=workers,
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
        "n_runs_per_search",  # number of random iterations
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
        "now_str",  # naming scheme
        "parallel_search",  # run with & at the end instead of ; to run in subshells
        "ipc",  # run with & at the end instead of ; to run in subshells
        "start_index",  # ignore the first runs, to continue a cartesian or sequential exploration for instance
        "use_transformer",  # defaults to True
        "use_tmpdir",  # use SLURM_TMPDIR and copy files to outdir after
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
    conf["now_str"] = now_str()
    infra = conf.get("infra", "mila")
    parallel_search = conf.get("parallel_search", False)
    start_index = conf.get("start_index", 0)
    use_transformer = conf.get("use_transformer", True)
    template_str = load_template(infra)
    use_tmpdir = conf.get("use_tmpdir", False)
    outdir = None
    dev = "dev" in conf and conf["dev"]

    home = os.environ["HOME"]

    if use_tmpdir:
        outdir = str(conf["outdir"])
        Path(outdir).resolve().mkdir(parents=True, exist_ok=True)
        conf["outdir"] = "$SLURM_TMPDIR"

    # run n_search jobs
    printlines()
    intel_str = ""
    run_idx = start_index
    for i in range(conf.get("n_search", 1)):
        print("\nJOB", i)
        # use a different ipc address for each run
        if use_transformer:
            ipcf, ipcb = ipc_addresses()
            conf["ipc"] = {"frontend": ipcf, "backend": ipcb}

        # fill template
        if infra == "mila":
            job_str = fill_mila_template(template_str, conf)
        elif infra == "beluga":
            job_str = fill_beluga_template(template_str, conf)
        elif infra == "intel":
            job_str = fill_intel_template(template_str, conf)
        else:
            raise ValueError("Unknown infra " + str(infra))

        # sample params
        opts = sample_search_conf(conf, run_idx)

        # specify server frontend
        if use_transformer:
            opts["INFERENCE_SERVER_ADDRESS"] = f'"{ipcf}"'

        # convert params to string command-line args
        hydra_args = get_hydra_args(opts, RANDOM_SEARCH_SPECIFIC_PARAMS)

        # do n_runs_per_search simulations per job
        for k in range(conf.get("n_runs_per_search", 1)):

            # echo commandlines run in job
            if not dev:
                job_str += f"\necho 'python run.py {hydra_args}'\n"

            command_suffix = "&\nsleep 5;\n" if parallel_search else ";\n"
            # intel doesn't have a log file so let's make one
            if infra == "intel":
                job_out = Path(home) / f"job_logs"
                job_out.mkdir(exist_ok=True)
                job_out = job_out / f"{now_str()}.out"
                print("Job logs:", str(job_out))
                command_suffix = f" &> {str(job_out)} {command_suffix}"

            # append run command
            job_str += "\n{}{}".format("python run.py" + hydra_args, command_suffix)
            run_idx += 1
            # sample next params
            opts = sample_search_conf(conf, run_idx)
            # specify server frontend
            if use_transformer:
                opts["INFERENCE_SERVER_ADDRESS"] = f'"{ipcf}"'

            # convert params to string command-line args
            hydra_args = get_hydra_args(opts, RANDOM_SEARCH_SPECIFIC_PARAMS)

        # output in slurm_tmpdir and move zips to original outdir specified
        if use_tmpdir and infra != "intel":
            # data  needs to be zipped for it to be transferred
            assert conf["zip_outdir"]
            job_str += f"\ncp $SLURM_TMPDIR/*.zip {outdir}"

        # create temporary sbatch file
        tmp = Path(tempfile.NamedTemporaryFile(suffix=".sh").name)
        # give somewhat meaningful name to t
        tmp = tmp.parent / (Path(conf.get("outdir", "")).name + "_" + tmp.name)
        if not dev:
            with tmp.open("w") as f:
                f.write(job_str)

        # sbatch or bash execution
        if infra in {"beluga", "mila"}:
            command = f"sbatch {str(tmp)}"
        elif infra == "intel":
            command = f"bash {str(tmp)}"

        # dev-mode: don't actually run the command
        if dev:
            print("\n>>> ", command, end="\n\n")
            print(str(tmp))
            print("." * 50)
            print(job_str)
        else:
            # not dev-mode: run it!
            _ = subprocess.call(command.split(), cwd=home)
            time.sleep(0.5)

        # prints
        print()
        printlines()


if __name__ == "__main__":
    main()
