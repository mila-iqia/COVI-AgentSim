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
from covid19sim.utils.utils import parse_search_configuration
from collections import defaultdict

SAMPLE_KEYS = {"list", "uniform", "range", "cartesian", "sequential"}


class RandomSearchError(Exception):
    pass


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

    if not (hydra_configs / "search" / f"{exp_file}.yaml").exists():
        raise RandomSearchError(f"Unknown exp_file {exp_file}")

    weights_dir = conf.get("weights_dir", "None")
    for tm in tracing_methods:
        weights = conf.get("weights", None)
        if "oracle" in tm or "transformer" in tm:
            if weights is None and ">" not in tm:
                raise RandomSearchError(
                    f"Unknown {tm} weights. Please specify '>' or 'weights: ...'"
                )
            elif ">" in tm:
                if not Path(weights_dir).exists():
                    raise RandomSearchError(
                        "No 'weights' specified and unknown 'weights_dir' {}".format(
                            weights_dir
                        )
                    )
                w = tm.split(">")[-1].strip()
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

    partition = conf.get("partition", "main")
    cpu = conf.get("cpus", 6)
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

    use_transformer = str(conf.get("use_transformer", True)).lower()
    workers = cpu - 1
    if "%j.out" not in slurm_log:
        slurm_log = str(Path(slurm_log).resolve() / "covi-slurm-%j.out")
        if not Path(slurm_log).parent.exists() and not conf.get("dev"):
            Path(slurm_log).parent.mkdir(parents=True)

    use_server = str(use_transformer and conf.get("USE_INFERENCE_SERVER", True)).lower()

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

    use_server = str(use_transformer and conf.get("USE_INFERENCE_SERVER", True)).lower()
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


@hydra.main(config_path="../configs/search/config.yaml", strict=False)
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
        "use_server",  # defaults to True
        "use_tmpdir",  # use SLURM_TMPDIR and copy files to outdir after
        "test_capacity",  # change TEST_TYPES.lab.capacity to that value
        "weights_dir",  # where are the weights
        "base_dir",  # output dir will be base_dir/intervention
    }

    # move back to original directory because hydra moved
    os.chdir(hydra.utils.get_original_cwd())

    # get command-line arguments as native dict
    overrides = parse_search_configuration(conf)

    # load experimental configuration
    # override with exp_file=<X>
    # where <X> is in configs/search and is ".yaml"
    conf = load_search_conf(
        Path(__file__).resolve().parent.parent
        / "configs"
        / "search"
        / (overrides.get("exp_file", "experiment") + ".yaml")
    )
    # override experimental parametrization with the commandline conf
    conf.update(overrides)
    check_conf(conf)

    # -------------------------------------
    # -----  Compute Specific Values  -----
    # -------------------------------------

    conf["n_runs_per_search"] = conf.get("n_runs_per_search", 1)

    if conf.get("n_search") == -1:
        total_runs = compute_n_search(conf)
        conf["n_search"] = total_runs // conf["n_runs_per_search"]
    else:
        total_runs = conf["n_runs_per_search"] * conf["n_search"]

    if total_runs % conf["n_runs_per_search"] != 0:
        raise RandomSearchError(
            "n_search ({}) is not divisible by n_runs_per_epoch ({})".format(
                total_runs, conf["n_runs_per_search"]
            )
        )

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

    # run n_search jobs
    printlines()
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

        # do n_runs_per_search simulations per job
        for k in range(conf.get("n_runs_per_search", 1)):

            opts = sample_search_conf(conf, run_idx)
            # specify server frontend

            use_transformer = opts["intervention"].split(">")[0].strip() in {
                "oracle",
                "transformer",
            }
            use_server = use_transformer and opts.get("USE_INFERENCE_SERVER", True)

            if use_transformer:
                if "weights" not in opts:
                    if ">" not in opts["intervention"]:
                        raise RandomSearchError("Unknown weights for transformer")
                    weights_name = opts["intervention"].split(">")[-1].strip()
                    opts["weights"] = str(Path(opts["weights_dir"]) / weights_name)
                if ">" in opts["intervention"]:
                    opts["intervention"] = (
                        opts["intervention"].split(">")[0].strip()
                    )

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

            if not opts.get("outdir"):
                extension = ""
                if opts["intervention"] == "transformer":
                    extension = Path(opts["weights"]).name
                opts["outdir"] = str(
                    Path(opts["base_dir"]) / (opts["intervention"] + extension)
                )

            if use_tmpdir:
                outdir = str(opts["outdir"])
                if not dev:
                    Path(outdir).resolve().mkdir(parents=True, exist_ok=True)
                opts["outdir"] = "$SLURM_TMPDIR"

            if opts["intervention"] == "no_intervention":
                opts["INTERVENTION_DAY"] = -1

            # convert params to string command-line args
            hydra_args = get_hydra_args(opts, RANDOM_SEARCH_SPECIFIC_PARAMS)
            if opts.get("test_capacity") is not None:
                hydra_args += f" TEST_TYPES.lab.capacity={opts.get('test_capacity')}"

            # echo commandlines run in job
            if not dev:
                job_str += f"\necho 'python run.py {hydra_args}'\n"

            command_suffix = "&\nsleep 5;\n" if parallel_search else ";\n"
            # intel doesn't have a log file so let's make one
            if infra == "intel":
                job_out = Path(home) / f"job_logs"
                if not dev:
                    job_out.mkdir(exist_ok=True)
                job_out = job_out / f"{now_str()}.out"
                print("Job logs:", str(job_out))
                command_suffix = f" &> {str(job_out)} {command_suffix}"

            # append run command
            job_str += "\n{}{}".format("python run.py" + hydra_args, command_suffix)
            run_idx += 1
            # sample next params

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
            print(sampled_str.format(**{k: opts[k] for k in sampled_keys}))

        # prints
        print()
        printlines()


if __name__ == "__main__":
    main()
