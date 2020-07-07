print("Loading imports...", end="", flush=True)
import math
import traceback
from collections import defaultdict
from pathlib import Path
from time import time
import pickle
import wandb
import os
import shutil

import hydra
from omegaconf import OmegaConf

import covid19sim.plotting.plot_jellybeans as jellybeans
import covid19sim.plotting.plot_pareto_adoption as pareto_adoption
import covid19sim.plotting.plot_presymptomatic as presymptomatic
from covid19sim.plotting.utils.extract_data import get_all_data

print("Ok.")
HYDRA_CONF_PATH = Path(__file__).parent.parent / "configs" / "plot"


def sizeof(path, suffix="B"):
    num = path.stat().st_size
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Y", suffix)


def help(all_plots):
    print("Available plotting options:")
    print("    * " + "\n    * ".join(all_plots.keys()))
    print("    * all")
    print('    * "[opt1, opt2, ...]" (<- note, lists are stringified)')
    print()
    print('Use exclude="[opt1, opt2, ...]" with `all` to run all plots but those')
    print()
    print("python main.py path=./app_adoption plot=pareto_adoption")
    print('python main.py plot=all exclude="[pareto_adoption]" # (<- note the "")')


def print_header(plot):
    print()
    print("#" * 60)
    print("##" + " " * 56 + "##")
    print(
        "##"
        + " " * math.floor((56 - len(plot)) / 2)
        + plot
        + " " * math.ceil((56 - len(plot)) / 2)
        + "##"
    )
    print("##" + " " * 56 + "##")
    print("#" * 60)


def print_footer():
    print("#" * 60)
    print()


def check_data(data):
    # TODO check all pkls exist
    # TODO check 1 pkl is loadable (3.7 vs 3.8)
    # TODO check all methods have same seeds and same comparisons
    pass


def summarize_configs(all_paths):
    ignore_keys = {"outdir", "GIT_COMMIT_HASH", "DAILY_TARGET_REC_LEVEL_DIST"}
    total_pkls = sum(len(mv) for mv in all_paths.values())
    print(
        "Found {} methods and {} pkls:\n    {}".format(
            len(all_paths),
            total_pkls,
            "\n    ".join(
                [
                    "{} ({})".format(Path(mk).name, len(mv))
                    for mk, mv in all_paths.items()
                ]
            ),
        )
    )
    confs = defaultdict(set)
    for mk, mv in all_paths.items():
        for rk, rv in mv.items():
            for ck, cv in rv["conf"].items():
                if ck in ignore_keys:
                    continue
                confs[ck].add(str(cv))
    print(
        "Varying parameters are:\n    "
        + "\n    ".join([f"{k}: {v}" for k, v in confs.items() if len(v) > 1])
    )


def get_model(conf, mapping):
    if conf["RISK_MODEL"] == "":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "unmitigated_norm"
        return "unmitigated"

    if conf["RISK_MODEL"] == "digital":
        if conf["TRACING_ORDER"] == 1:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "bdt1_norm"
            return "bdt1"
        elif conf["TRACING_ORDER"] == 2:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "bdt2_norm"
            return "bdt2"
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
        model = Path(conf["TRANSFORMER_EXP_PATH"]).name
        if model not in mapping:
            print(
                "Warning: unknown model name {}. Defaulting to `transformer`".format(
                    model
                )
            )
        model_name = (
            mapping.get(model, "transformer")
            + "-"
            + str(conf.get("REC_LEVEL_THRESHOLDS"))
        )
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            model_name = model_name + "_norm"
        return model_name

    if conf["RISK_MODEL"] == "heuristicv1":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "heuristicv1_norm"
        return "heuristicv1"

    if conf["RISK_MODEL"] == "heuristicv2":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "heuristicv2_norm"
        return "heuristicv2"

    raise ValueError("Unknown RISK_MODEL {}".format(conf["RISK_MODEL"]))


def map_conf_to_models(all_paths, plot_conf):
    new_data = defaultdict(lambda: defaultdict(dict))
    key = plot_conf["compare"]
    for mk, mv in all_paths.items():
        for rk, rv in mv.items():
            sim_conf = rv["conf"]
            compare_value = str(sim_conf[key])
            model = get_model(sim_conf, plot_conf["model_mapping"])
            new_data[model][compare_value][rk] = rv
    return dict(new_data)


def parse_options(conf, all_plots):
    options = {plot: {} for plot in all_plots}
    for arg in conf:
        for plot in all_plots:
            if arg.startswith(plot):
                opt_key = arg.replace(plot + "_", "")
                opt_value = conf[arg]
                options[plot][opt_key] = opt_value
    return dict(options)


@hydra.main(config_path=str(HYDRA_CONF_PATH.resolve() / "config.yaml"), strict=False)
def main(conf):
    # --------------------------------
    # -----  Parse Command-Line  -----
    # --------------------------------
    all_plots = {
        "pareto_adoption": pareto_adoption,
        "jellybeans": jellybeans,
        "presymptomatic": presymptomatic,
    }

    conf = OmegaConf.to_container(conf)
    options = parse_options(conf, all_plots)

    if conf["use_wandb"]:
        wandb.init(project="COVI")

    root_path = conf.get("path", ".")
    plot_path = os.path.join(root_path, "plots")

    root_path = Path(root_path).resolve()
    plot_path = Path(plot_path).resolve()

    if plot_path.exists():
        shutil.rmtree(plot_path)
    os.makedirs(plot_path)
    assert plot_path.exists()
    cache_path = root_path / "cache.pkl"

    # -------------------
    # -----  Help?  -----
    # -------------------
    if "help" in conf:
        help(all_plots)
        return

    # --------------------------
    # -----  Select Plots  -----
    # --------------------------
    plots = conf["plot"]
    if plots == "all":
        plots = list(all_plots.keys())
    if not isinstance(plots, list):
        plots = [plots]

    assert all(p in all_plots for p in plots), "Allowed plots are {}".format(
        "\n   ".join(all_plots.keys())
    )

    # --------------------------------------
    # -----  Select tracker data keys  -----
    # --------------------------------------
    keep_pkl_keys = set()
    if "pareto_adoption" in plots:
        keep_pkl_keys.update(
            [
                "intervention_day",
                "outside_daily_contacts",
                "effective_contacts_since_intervention",
                "intervention_day",
                "cases_per_day",
                "n_humans",
                "generation_times",
                "humans_state",
                "humans_intervention_level",
                "humans_rec_level",
            ]
        )
    if "jellybeans" in plots:
        keep_pkl_keys.update(
            ["humans_intervention_level", "humans_rec_level", "intervention_day"]
        )
    if "presymptomatic" in plots:
        keep_pkl_keys.update(["human_monitor"])

    # ------------------------------------
    # -----  Load pre-computed data  -----
    # ------------------------------------
    cache = None
    use_cache = cache_path.exists() and conf.get("use_cache", True)
    if use_cache:
        try:
            print(
                "Using cached data ({}): {}...".format(
                    sizeof(cache_path), str(cache_path)
                )
            )
            with cache_path.open("rb") as f:
                cache = pickle.load(f)

            # check that the loaded data contains what is required by current `plots`
            if "plots" not in cache or not all(p in cache["plots"] for p in plots):
                print(
                    "Missing some data for plots {} in cache.pkl".format(
                        ", ".join([p for p in plots if p not in cache["plots"]])
                    )
                )
                use_cache = False
            else:
                data = cache["data"]
                check_data(data)

        except Exception:
            print(
                "{}\n{}\n{}\n\nCould not load {}. Recomputing data.".format(
                    "*" * 30, traceback.format_exc(), "*" * 30, str(cache_path),
                )
            )
    if not use_cache:
        # --------------------------
        # -----  Compute Data  -----
        # --------------------------
        print("Reading configs from {}:".format(str(root_path)))
        rtime = time()
        all_data = get_all_data(
            root_path, keep_pkl_keys, conf.get("multithreading", False)
        )
        print("\nDone in {:.2f}s.\n".format(time() - rtime))
        summarize_configs(all_data)
        data = map_conf_to_models(all_data, conf)
        check_data(data)

        # -------------------------------
        # -----  Dump if requested  -----
        # -------------------------------
        if conf.get("dump_cache", True):
            print("Dumping cache...", end="", flush=True)
            t = time()
            with cache_path.open("wb") as f:
                if cache is None:
                    cache = {"plots": plots, "data": data}
                else:
                    cache["plots"] = list(set(cache["plots"] + plots))
                    cache["data"] = {**cache["data"], **data}
                pickle.dump(cache, f)
            print("Done in {}s ({})".format(int(time() - t), sizeof(cache_path)))

    for plot in plots:
        func = all_plots[plot].run
        print_header(plot)
        try:
            # -------------------------------
            # -----  Run Plot Function  -----
            # -------------------------------
            func(data, plot_path, conf["compare"], conf["use_wandb"], **options[plot])
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                print("Interrupting.")
                break
            else:
                print("** ERROR **")
                print(traceback.format_exc())
                print("*" * len(str(e)))
                print("Ignoring " + plot)
        print_footer()


if __name__ == "__main__":
    main()
