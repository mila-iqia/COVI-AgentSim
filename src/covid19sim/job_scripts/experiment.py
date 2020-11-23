import datetime
import itertools
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
import ast
import json

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from covid19sim.plotting.utils import env_to_path
from covid19sim.utils.utils import parse_search_configuration, is_app_based_tracing_intervention, NpEncoder

SAMPLE_KEYS = {"list", "uniform", "range", "cartesian", "sequential", "chain"}
HYDRA_CONF_PATH = Path(__file__).parent.parent / "configs/exp/config.yaml"
np.random.seed(seed=0)

class RandomSearchError(Exception):
    pass


class SampleWithMemory(object):
    """
    Samples parameters such that non-cartesian parameters are kept in memory until all cartesian samples are used.

    Args:
        conf (dict): yaml configuration of the experiment
    """
    def __init__(self, conf):
        self.rng = np.random.RandomState(seed=conf['SAMPLE_WITH_MEMORY_SEED'])
        self.memory_parameters = {}
        self.other_parameters = []
        self.max_idx = 1
        for key, value in conf.items():
            if (
                isinstance(value, dict)
                and "sample" in value
            ):

                assert value['sample'] not in ['range', 'list'], "Sampling from range and list hasn't been fully considered. It might be wrong."
                if value.get('memory', False):
                    self.memory_parameters[key] = value
                else:
                    if value['sample'] == "uniform":
                        n = value['n']
                    else:
                        assert value['sample'] not in ['sequential', 'chain'], "Sequential and Chain sampling with memory not implemented"
                        assert value['sample'] == "cartesian", "expected sampling type: cartesian"
                        n = len(value['from'])
                    self.max_idx *= n
                    self.other_parameters.append((key, value)) # all combinations will be taken

        self._idx = self.max_idx

    def get_new_combinations(self):
        """
        """
        sampled_values = []
        for key, value in self.other_parameters:
            if value['sample'] == "uniform":
                xs  = self.rng.uniform(*value["from"], size=value['n'])
                sampled_values.append([(key, x) for x in xs])
            elif value['sample'] == "cartesian":
                sampled_values.append([(key, v) for v in value['from']])

        combinations = list(itertools.product(*sampled_values))
        return combinations


    def __call__(self, exp, run_idx=None):
        if self._idx == self.max_idx:
            self.current_conf = deepcopy(exp)
            self.current_conf.update(sample_search_conf(self.memory_parameters))
            self.current_combinations = self.get_new_combinations()
            self._idx = 0

        conf = deepcopy(self.current_conf)
        for k,v in self.current_combinations[self._idx]:
            conf[k] = v
        # conf.update(sample_cartesians(self.cartesians, exp, self._idx))
        self._idx += 1
        keys_to_remove = []
        for key in conf.keys():
            if "SAMPLE_WITH_MEMORY" in key:
                keys_to_remove.append(key)
        _ = [conf.pop(k, None) for k in keys_to_remove]
        return conf


def first_key(d):
    """get the first key of a dict"""
    return list(d.keys())[0]


def first_value(d):
    """get the first value of a dict"""
    return list(d.values())[0]


def get_extension(x):
    """
    Map a key, value tuple to a string to create the folder name in base_dir
    """
    k, v = x
    if k == "REC_LEVEL_THRESHOLDS":
        return "".join(map(str, (v if not isinstance(v, str) else ast.literal_eval(v))))
    return str(v)


def get_model(conf):
    if conf["RISK_MODEL"] == "":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "unmitigated_normed"
        return "unmitigated"

    if conf["RISK_MODEL"] == "digital":
        if conf["TRACING_ORDER"] == 1:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "binary_digital_tracing_order_1_normed"
            return "binary_digital_tracing_order_1"
        elif conf["TRACING_ORDER"] == 2:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "binary_digital_tracing_order_2_normed"
            return "binary_digital_tracing_order_2"
        else:
            raise ValueError(
                "Unknown binary digital tracing order: {}".format(conf["TRACING_ORDER"])
            )

    if conf["RISK_MODEL"] == "transformer":
        if conf["USE_ORACLE"]:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "oracle_normed"
            return "oracle"
        # FIXME this won't work if the run used the inference server
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "transformer_normed"
        return "transformer"

    if conf["RISK_MODEL"] == "heuristicv1":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "heuristicv1_normed"
        return "heuristicv1"

    if conf["RISK_MODEL"] == "heuristicv2":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "heuristicv2_normed"
        return "heuristicv2"

    raise ValueError("Unknown RISK_MODEL {}".format(conf["RISK_MODEL"]))


def normalize(opts):
    if "normalization_folder" not in opts:
        return opts

    folder = opts["normalization_folder"]
    run = opts[folder]

    sim_configs = Path(__file__).resolve().parent.parent / "configs" / "simulation"
    run_yaml = sim_configs / folder / f"{run}.yaml"

    if not run_yaml.exists():
        raise RandomSearchError(
            f"Unknown normalized run: \nFolder: {folder}\nFile: {run_yaml.name}"
        )

    with run_yaml.open("r") as f:
        run_conf = yaml.safe_load(f)

    opts["intervention"] = get_model(run_conf)
    if "TRANSFORMER_EXP_PATH" in run_conf:
        weights = Path(run_conf["TRANSFORMER_EXP_PATH"]).name
        opts["intervention"] += f">{weights}"
    return opts


def check_conf(conf):
    tracing_methods = []
    hydra_configs = Path(__file__).resolve().parent.parent / "configs"
    exp_file = conf.get("exp_file", "experiment")
    use_tmpdir = conf.get("use_tmpdir")
    infra = conf.get("infra")
    zip_outdir = conf.get("zip_outdir")
    for k, v in conf.items():
        if isinstance(v, dict) and "sample" in v:
            if v["sample"] not in SAMPLE_KEYS:
                raise RandomSearchError(
                    "Unknown sampling procedure {} for {}".format(v["sample"], k)
                )
            if "from" not in v:
                raise RandomSearchError(f"No 'from' key for {k}")
            if v["sample"] == "cartesian" and not isinstance(v["from"], list):
                raise RandomSearchError(f"'from' field for {k} should be a list")

        if k == "intervention":
            if isinstance(v, dict) and "sample" in v:
                tracing_methods += v["from"]
            else:
                tracing_methods.append(v)

        elif k == "tune":
            if conf.get("use_tmpdir", False) is not False:
                raise RandomSearchError("Cannot use 'tune' and use_tmpdir:true")

        elif k == "run_type":
            if not (hydra_configs / "simulation" / "run_type" / f"{v}.yaml").exists():
                raise RandomSearchError(f"Unknown run_type {v}")

    if not (hydra_configs / "experiment" / f"{exp_file}.yaml").exists():
        raise RandomSearchError(f"Unknown exp_file {exp_file}")

    weights_dir = conf.get("weights_dir", "None")
    for tm in tracing_methods:
        weights = conf.get("weights", None)
        if isinstance(tm, dict):
            model = first_key(tm)
            if weights is None and "weights" not in tm[model]:
                raise RandomSearchError(
                    f"Unknown {tm[model]} weights. Please specify '>' or 'weights: ...'"
                )
            elif "weights" in tm[model]:
                if not Path(weights_dir).exists():
                    raise RandomSearchError(
                        "No 'weights' specified and unknown 'weights_dir' {}".format(
                            weights_dir
                        )
                    )
                w = tm[model]["weights"]
                weights = Path(weights_dir) / w
            elif weights is not None:
                weights = Path(weights)

            if not weights.exists():
                raise RandomSearchError("Cannot find weights {}".format(str(weights)))
            else:
                transformer_best = weights / "Weights" / "best.ckpt"
                transformer_config = weights / "Configurations" / "train_config.yml"
                if not transformer_best.exists():
                    raise RandomSearchError(
                        "Cannot find weights {}".format(transformer_best)
                    )
                if not transformer_config.exists():
                    raise RandomSearchError(
                        "Cannot find train config {}".format(transformer_config)
                    )
    if use_tmpdir and infra != "intel" and not zip_outdir:
        raise RandomSearchError(
            "zip_outdir must be true when using tmpdir (use_tmpdir)"
        )


def compute_n_search(conf):
    """
    Compute the number of searchs to do if using -1 as n_search and using
    cartesian search

    Args:
        conf (dict): exprimental configuraiton

    Raises:
        RandomSearchError: Cannot be called if no cartesian or sequential field

    Returns:
        int: size of the cartesian product or length of longest sequential field
    """
    samples = defaultdict(list)
    for k, v in conf.items():
        if not isinstance(v, dict) or "sample" not in v:
            continue
        samples[v["sample"]].append(v)

    if "cartesian" in samples:
        total = 1
        for s in samples["cartesian"]:
            total *= len(s["from"])
        return total
    if "sequential" in samples:
        total = max(map(len, [s["from"] for s in samples["sequential"]]))
        return total
    if "chain" in samples:
        total = sum(map(len, [s["from"] for s in samples["chain"]]))
        return total

    raise RandomSearchError(
        "Used n_search=-1 without any field being 'cartesian' or 'sequential'"
    )


def now_str():
    """
    20200608_125339_353416
    """
    now = str(datetime.datetime.now())
    now = now.replace("-", "").replace(":", "").replace(" ", "_").replace(".", "_")
    return now


def sample_param(sample_dict, rng=np.random):
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

    if sample_dict["sample"] == "chain":
        assert isinstance(
            sample_dict["from"], list
        ), "{}'s `from` field MUST be a list, found {}".format(
            sample_dict["sample"], sample_dict["from"]
        )
        return "__chain__"

    if sample_dict["sample"] == "range":
        return rng.choice(np.arange(*sample_dict["from"]))

    if sample_dict["sample"] == "list":
        return rng.choice(sample_dict["from"])

    if sample_dict["sample"] == "uniform":
        return rng.uniform(*sample_dict["from"])

    raise ValueError("Unknown sample type in dict " + str(sample_dict))


def load_search_conf(path):
    """
    Load a yaml file in `path`

    Args:
        path (str | pathlib.Path): path to yaml file

    Returns:
        any: python native variable loaded by PyYaml
    """
    print(path)
    print()
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
        cartesian_keys (list): keys in the experimental configuration that
            are to be used in the full cartesian product
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


def sample_chains(chain_keys, exp, idx):
    """
    Returns the `idx`th item in the chain of all chain keys to be sampled.

    Args:
        chain_keys (list): keys in the experimental configuration
            that are to be used in the full chain
        exp (dict): experimental configuration
        idx (int): index of the current sample

    Returns:
        dict: sampled point in the cartesian space (with keys = chain_keys)
    """
    conf = {}
    chain_values = [[(key, value) for value in exp[key]["from"]] for key in chain_keys]
    chain = list(itertools.chain(*chain_values))
    k, v = chain[idx % len(chain)]
    conf[k] = v
    if exp[k].get("normalized"):
        conf["normalization_folder"] = k
    return conf


def sample_sequentials(sequential_keys, exp, idx):
    """
    Samples sequentially from the "from" values specified in each key
    of the experimental configuration which have sample == "sequential"
    Unlike `cartesian` sampling, `sequential` sampling iterates *independently*
    over each keys

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
        idx (int): experiment's idx in the sampling procedure (useful in case a key
            should be sampled in a cartesian or sequential manner)

    Returns:
        dict: sampled configuration
    """
    conf = {}
    cartesians = []
    sequentials = []
    chains = []
    for k, v in exp.items():
        candidate = sample_param(v)
        if candidate == "__cartesian__":
            cartesians.append(k)
        elif candidate == "__sequential__":
            sequentials.append(k)
        elif candidate == "__chain__":
            chains.append(k)
        else:
            conf[k] = candidate
    if sequentials:
        conf.update(sample_sequentials(sequentials, exp, idx))
    if chains:
        conf.update(sample_chains(chains, exp, idx))
    if cartesians:
        conf.update(sample_cartesians(cartesians, exp, idx))
    return conf


def load_template(infra):
    """
    Get the template string to format according to arguments

    Returns:
        str: template string full of "{variable_name}"
    """
    base = Path(__file__).resolve().parent
    if infra == "mila":
        with (base / "mila_sbatch_template.sh").open("r") as f:
            return f.read()
    if infra == "intel":
        with (base / "intel_template.sh").open("r") as f:
            return f.read()
    if infra == "beluga":
        with (base / "beluga_sbatch_template.sh").open("r") as f:
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
    email = conf.get('email_id', "")

    partition = conf.get("partition", "main")
    cpu = conf.get("cpus", 6)
    # cpu constraints in long partition
    if partition == "long":
        cpu = min(cpu, 4)

    mem = conf.get("mem", 16)
    gres = conf.get("gres", "")
    time = str(conf.get("time", "4:00:00"))
    slurm_log = conf.get(
        "slurm_log", conf.get("base_dir", f"/network/tmp1/{user}/covi-slurm-%j.out")
    )
    env_name = conf.get("env_name", "covid")
    weights = conf.get("weights")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))
    ipc = conf.get("ipc", {"frontend": "", "backend": ""})

    use_transformer = conf.get("use_transformer", True)
    workers = cpu - 1
    if "%j.out" not in slurm_log:
        slurm_log = str(Path(slurm_log).resolve() / "covi-slurm-%j.out")
        if not Path(slurm_log).parent.exists() and not conf.get("dev"):
            Path(slurm_log).parent.mkdir(parents=True)

    use_server = str(use_transformer and conf.get("USE_INFERENCE_SERVER", False)).lower()

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
                    "  {:10}: {}".format("use_server", use_server),
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
    email = f"#SBATCH --mail-user={email}"
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
        use_server=use_server,
        workers=workers,
        email=email
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
    email = conf.get('email_id', "")

    cpu = conf.get("cpus", 4)
    mem = conf.get("mem", 12)
    time = str(conf.get("time", "2:50:00"))
    slurm_log = conf.get(
        "slurm_log", conf.get("base_dir", f"/scratch/{user}/covi-slurm-%j.out")
    )
    if "%j.out" not in slurm_log:
        slurm_log = str(Path(slurm_log).resolve() / "covi-slurm-%j.out")
        if not Path(slurm_log).parent.exists() and not conf.get("dev"):
            Path(slurm_log).parent.mkdir(parents=True)
    env_name = conf.get("env_name", "covid")
    weights = conf.get("weights")
    code_loc = conf.get("code_loc", str(Path(home) / "simulator/src/covid19sim/"))
    ipc = conf.get("ipc", {"frontend": "", "backend": ""})
    use_transformer = conf.get("use_transformer", True)

    use_server = str(use_transformer and conf.get("USE_INFERENCE_SERVER", False)).lower()
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
                    "  {:10}: {}".format("use_server", use_server),
                    "  {:10}: {}".format("workers", workers),
                ]
            )
        )

    cpu = f"#SBATCH --cpus-per-task={cpu}"
    mem = f"#SBATCH --mem={mem}GB"
    time = f"#SBATCH --time={time}"
    email = f"#SBATCH --mail-user={email}"
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
        use_server=use_server,
        workers=workers,
        email=email
    )


def get_hydra_args(opts, exclude=set()):
    hydra_args = ""
    for k, v in opts.items():
        if k not in exclude:
            if isinstance(v, list):
                v = f'"{v}"'
            hydra_args += f" {k}={v}"
    return hydra_args


def printlines():
    print("=" * 80)
    print("=" * 80)

HYDRA_CONF_PATH = "../configs/experiment/config.yaml"
@hydra.main(config_path=HYDRA_CONF_PATH, strict=False)
def main(conf: DictConfig) -> None:
    """
                HOW TO USE

    $ python experiment.py exp_file=experiment n_search=20

    add `dev=True` to just see the commands that would be run, without
    running them

    NOTE: ALL parameters used in run.py may be overwritten from this commandline.
    For instance you can change init_fraction_sick

    $ python experiment.py exp_file=experiment n_search=20 init_fraction_sick=0.1

    NOTE: you may also pass arguments overwriting the default `sbatch` job's
    parametrization like partition, gres, code_loc (=where is the simulator's code),
    env_name (= what conda env to load). For instance:

    $ python experiment.py partition=unkillable gres=gpu:1 env_name=covid-env\
                              n_search=20 init_fraction_sick=0.1

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
        "weights",  # where to find the transformer's weights
        "infra",  # using Mila or Intel cluster?
        "now_str",  # naming scheme
        "parallel_search",  # run with & at the end instead of ; to run in subshells
        "ipc",  # run with & at the end instead of ; to run in subshells
        "start_index",  # ignore the first runs, to continue an exploration for instance
        "use_transformer",  # defaults to True
        "use_server",  # defaults to True
        "use_tmpdir",  # use SLURM_TMPDIR and copy files to outdir after
        "weights_dir",  # where are the weights
        "base_dir",  # output dir will be base_dir/tracing_method
        "normalization_folder",  # if this is a normalization run
        "exp_name",  # folder name in base_dir => base_dir/exp_name/method/...
        "email_id", # email id where you can receive notifications regarding jobs (began, completed, failed)
    }

    # move back to original directory because hydra moved
    os.chdir(hydra.utils.get_original_cwd())

    # get command-line arguments as native dict
    overrides = parse_search_configuration(conf)

    # load experimental configuration
    # override with exp_file=<X>
    # where <X> is in configs/exp and is ".yaml"
    exp_file_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "experiment"
        / (overrides.get("exp_file", "randomization") + ".yaml")
    )
    conf = load_search_conf(exp_file_path)
    # override experimental parametrization with the commandline conf
    conf.update(overrides)
    check_conf(conf)

    for k in ["code_loc", "base_dir", "outdir", "weights_dir"]:
        if k in conf and conf[k]:
            conf[k] = env_to_path(conf[k])

    # -------------------------------------
    # -----  Compute Specific Values  -----
    # -------------------------------------

    conf["n_runs_per_search"] = conf.get("n_runs_per_search", 1)

    if conf.get("n_search") == -1:
        total_runs = compute_n_search(conf) * conf.get('SAMPLE_WITH_MEMORY_ITERATIONS', 1)
        conf["n_search"] = total_runs // conf["n_runs_per_search"]
    else:
        total_runs = conf["n_runs_per_search"] * conf["n_search"]

    if total_runs % conf["n_runs_per_search"] != 0:
        raise RandomSearchError(
            "n_search ({}) is not divisible by n_runs_per_epoch ({})".format(
                total_runs, conf["n_runs_per_search"]
            )
        )

    if "exp_name" in conf:
        if "base_dir" in conf:
            conf["base_dir"] = str(Path(conf["base_dir"]) / conf["exp_name"])
            print(f"Running experiments in base_dir: {conf['base_dir']}")
        else:
            print(f"Ignoring 'exp_name' {conf['exp_name']} as no base_dir was provided")

    print(f"Running {total_runs} scripts")

    conf["now_str"] = now_str()
    infra = conf.get("infra", "mila")
    parallel_search = conf.get("parallel_search", False)
    start_index = conf.get("start_index", 0)
    template_str = load_template(infra)
    use_tmpdir = conf.get("use_tmpdir", False)
    outdir = None
    dev = "dev" in conf and conf["dev"]
    is_tune = conf.get("tune", False)
    sampled_keys = [k for k, v in conf.items() if isinstance(v, dict) and "sample" in v]
    sampled_str = "\n".join([f"  {k}: {{{k}}}" for k in sampled_keys])

    if is_tune and use_tmpdir:
        raise RandomSearchError("Cannot use tune and $SLURM_TMPDIR together")
    if use_tmpdir and not conf["outdir"]:
        raise RandomSearchError(
            "Using $SLURM_TPMDIR but no `outdir` has been specified"
        )

    home = os.environ["HOME"]
    copy_dest = conf["outdir"] if "outdir" in conf else conf["base_dir"]
    if not dev:
        Path(copy_dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(exp_file_path, Path(copy_dest) / exp_file_path.name)

    # run n_search jobs
    sample_conf = sample_search_conf
    if conf.get('SAMPLE_WITH_MEMORY', False):
        sample_conf = SampleWithMemory(conf)

    printlines()
    old_opts = set()
    run_idx = start_index
    for i in range(conf.get("n_search", 1)):
        print("\nJOB", i)
        ipcf, ipcb = None, None

        # fill template
        if infra == "mila":
            job_str = fill_mila_template(template_str, conf)
        elif infra == "beluga":
            job_str = fill_beluga_template(template_str, conf)
        elif infra == "intel":
            job_str = fill_intel_template(template_str, conf)
        else:
            raise ValueError("Unknown infra " + str(infra))

        # do n_runs_per_search simulations per job submission
        for k in range(conf.get("n_runs_per_search", 1)):
            skipped = False
            opts = sample_conf(conf, run_idx)
            opts = normalize(opts)
            run_idx += 1

            # rewrite APP_UPTAKE for non-tracing methods to avoid redundant experiments
            if not is_app_based_tracing_intervention(opts['intervention']):
                opts['APP_UPTAKE'] = -1

            opts_str = json.dumps(opts, sort_keys=True, cls=NpEncoder)
            if opts_str in old_opts:
                print("\n Ran this job already ... skipping!")
                skipped = True
                continue

            old_opts.add(opts_str)

            extension = ""
            # specify server frontend

            tracing_dict = None
            tracing_name = None
            if isinstance(opts.get("intervention", ""), dict):
                tracing_dict = first_value(opts["intervention"])
                tracing_name = first_key(opts["intervention"])

            use_transformer = (
                tracing_dict is not None
                and "weights" in tracing_dict
                and tracing_dict["weights"]
            )
            use_server = use_transformer and opts.get("USE_INFERENCE_SERVER", False)

            if use_transformer:
                # -------------------------
                # -----  Set Weights  -----
                # -------------------------

                if "weights" not in opts:
                    weights_name = tracing_dict["weights"]
                    weights_name = weights_name.strip()
                    opts["weights"] = str(Path(opts["weights_dir"]) / weights_name)

            if tracing_dict is not None:
                # Create folder name extension based on keys in tracing_method dict
                extensions = sorted(tracing_dict.items())

                extension = "_" + "_".join(map(get_extension, extensions))

                # Add tracing_method dict's keys and values to opts
                for k, v in tracing_dict.items():
                    if k != "weights":
                        if k in opts:
                            print(
                                "Warning, overriding opts[{}]={} to opts[{}]={}".format(
                                    k, opts[k], k, v
                                )
                            )
                        opts[k] = v

                # set true tracing_method
                opts["intervention"] = tracing_name

            # -----------------------------------------------------
            # -----  Inference Server / Transformer Exp Path  -----
            # -----------------------------------------------------
            if use_server:
                if ipcf is None:
                    ipcf, ipcb = ipc_addresses()
                opts["ipc"] = {"frontend": ipcf, "backend": ipcb}
                opts["INFERENCE_SERVER_ADDRESS"] = f'"{ipcf}"'
            else:
                if opts.get("USE_INFERENCE_SERVER") is not False:
                    opts["USE_INFERENCE_SERVER"] = False
                if use_transformer:
                    opts["TRANSFORMER_EXP_PATH"] = opts["weights"]

            # ----------------------------------------------
            # -----  Set outdir from basedir (if any)  -----
            # ----------------------------------------------
            if not opts.get("outdir"):
                opts["outdir"] = Path(opts["base_dir"]).resolve()
                opts["outdir"] = opts["outdir"] / (opts["intervention"] + extension)
                opts["outdir"] = str(opts["outdir"])

            opts["outdir"] = opts["outdir"]

            # --------------------------------
            # -----  Use SLURM_TMPDIR ?  -----
            # --------------------------------
            if use_tmpdir:
                outdir = str(opts["outdir"])
                if not dev:
                    Path(outdir).resolve().mkdir(parents=True, exist_ok=True)
                opts["outdir"] = "$SLURM_TMPDIR"

            # overwrite intervention day if no_intervention
            if opts["intervention"] == "no_intervention":
                opts["INTERVENTION_DAY"] = -1

            # convert params to string command-line args
            exclude = RANDOM_SEARCH_SPECIFIC_PARAMS
            if opts.get("normalization_folder"):
                exclude.add("intervention")
            hydra_args = get_hydra_args(opts, exclude)

            # echo commandlines run in job
            if not dev:
                job_str += f"\necho 'python run.py {hydra_args}'\n"

            command_suffix = "&\nsleep 5;\n" if parallel_search else ";\n"
            # intel doesn't have a log file so let's make one
            if infra == "intel":
                job_out = Path(home) / "job_logs"
                if not dev:
                    job_out.mkdir(exist_ok=True)
                job_out = job_out / f"{now_str()}.out"
                print("Job logs:", str(job_out))
                command_suffix = f" &> {str(job_out)} {command_suffix}"

            # append run command
            job_str += "\n{}{}".format("python run.py" + hydra_args, command_suffix)
            # sample next params

        if skipped:
            continue
        # output in slurm_tmpdir and move zips to original outdir specified
        if use_tmpdir and infra != "intel":
            # data  needs to be zipped for it to be transferred
            assert opts["zip_outdir"]
            job_str += f"\ncp $SLURM_TMPDIR/*.zip {outdir}"

        # create temporary sbatch file
        tmp = Path(tempfile.NamedTemporaryFile(suffix=".sh").name)
        # give somewhat meaningful name to t
        tmp = tmp.parent / (Path(opts.get("outdir", "")).name + "_" + tmp.name)
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
            print("In", opts["outdir"])
            print("With Sampled Params:")
            print(sampled_str.format(**{k: opts.get(k) for k in sampled_keys}))

        # prints
        print()
        printlines()


if __name__ == "__main__":
    main()
