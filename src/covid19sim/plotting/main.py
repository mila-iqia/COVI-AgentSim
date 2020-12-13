print("Loading imports...", end="", flush=True)
import math
import traceback
from collections import defaultdict
from pathlib import Path
from time import time
import dill
import pickle
import os
import shutil
import hydra
from omegaconf import OmegaConf
import random

import covid19sim.plotting.plot_jellybeans as jellybeans
import covid19sim.plotting.plot_pareto_adoption as pareto_adoption
import covid19sim.plotting.plot_presymptomatic as presymptomatic
import covid19sim.plotting.plot_reasons as reasons
import covid19sim.plotting.make_efficiency_table as efficiency
import covid19sim.plotting.plot_generation_time as generation_time
import covid19sim.plotting.plot_epi_table as epi_table
import covid19sim.plotting.plot_episim_series as episim_series
import covid19sim.plotting.plot_normalized_mobility_scatter as normalized_mobility
import covid19sim.plotting.plot_dalys as dalys
import covid19sim.plotting.plot_sensitivity as sensitivity
import covid19sim.plotting.plot_grid_sensitivity as grid_sensitivity
from covid19sim.plotting.utils import get_all_data


print("Ok.")
HYDRA_CONF_PATH = Path(__file__).resolve().parent.parent / "configs" / "plot"


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


def map_conf_to_models(all_paths, plot_conf):
    new_data = {}  # defaultdict(lambda: defaultdict(dict))
    key = plot_conf["compare"]
    for mk, mv in all_paths.items():
        for rk, rv in mv.items():
            sim_conf = rv["conf"]
            compare_value = str(sim_conf[key])
            model = Path(rk).parent.name
            if model not in new_data:
                new_data[model] = {}
            if compare_value not in new_data[model]:
                new_data[model][compare_value] = {}
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
        "reasons": reasons,
        "presymptomatic": presymptomatic,
        "efficiency": efficiency,
        "generation_time": generation_time,
        "epi_table": epi_table,
        "dalys": dalys,
        "episim_series": episim_series,
        "normalized_mobility": normalized_mobility,
        "sensitivity": sensitivity,
        "grid_sensitivity": grid_sensitivity
    }

    conf = OmegaConf.to_container(conf)
    options = parse_options(conf, all_plots)

    root_path = conf.get("path", ".")
    plot_path = os.path.join(root_path, "plots")

    root_path = Path(root_path).resolve()
    plot_path = Path(plot_path).resolve()

    if plot_path.exists() and conf.get("clear_plots"):
        shutil.rmtree(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    assert plot_path.exists()
    cache_path = root_path / "cache.pkl"

    # -------------------
    # -----  Help?  -----
    # -------------------
    if conf.get("help", False):
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

    plots = [p for p in plots if p not in conf.get("exclude", [])]

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
                "healthy_effective_contacts_since_intervention",
                "intervention_day",
                "cases_per_day",
                "n_humans",
                "generation_times",
                "humans_state",
                "humans_intervention_level",
                "humans_rec_level",
                "GLOBAL_MOBILITY_SCALING_FACTOR",
                "infection_monitor",
                "humans_quarantined_state"
            ]
        )
    if "dalys" in plots:
        # Same as pareto
        keep_pkl_keys.update(
            [
                "intervention_day",
                "human_has_app",
                "age_histogram",
                "n_humans",
                "intervention_day",
                "intervention",
                "human_monitor",
                "quarantine_monitor",
                "humans_rec_level",
                "humans_intervention_level",
                "humans_demographics",
                "work_hours"
            ]
        )
    if "efficiency" in plots:
        # Same as pareto
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
    if "reasons" in plots:
        keep_pkl_keys.update(["risk_attributes"])
    if "generation_time" in plots:
        keep_pkl_keys.update(["infection_monitor"])
    if "epi_table" in plots:
        keep_pkl_keys.update(["covid_properties", "generation_times", "daily_age_group_encounters", "age_histogram",
                              "r_0", "contacts"])

    if "episim_series" in plots:
        keep_pkl_keys.update([
            "recovered_stats",
            "cases_per_day",
            "ei_per_day",
            "s",
            "n_humans",
            "infectious_contact_patterns",
            "human_has_app",
            "intervention_day",
            "humans_state",
            "humans_quarantined_state"
        ])

    if (
        "normalized_mobility" in plots
        or "sensitivity" in plots
    ):
        keep_pkl_keys.update([
            "infection_monitor",
            "humans_state",
            "intervention_day",
            "n_humans",
            "humans_quarantined_state",
            "effective_contacts_since_intervention",
            "healthy_effective_contacts_since_intervention",
            "cases_per_day",
            "test_monitor",
            "humans_demographics"
        ])

    # ------------------------------------
    # -----  Load pre-computed data  -----
    # ------------------------------------
    cache = None
    use_cache = conf.get("use_cache", True)
    load_cache = conf.get("load_cache", use_cache)
    if load_cache:
        assert cache_path.exists(), f"Cache {str(cache_path)} doesn't exist."
    if use_cache:
        data = {}
        if load_cache:
            try:
                print(
                    "Using cached data ({}): {}...".format(
                        sizeof(cache_path), str(cache_path)
                    )
                )
                with cache_path.open("rb") as f:
                    cache = dill.load(f)

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
    else:
        # --------------------------
        # -----  Compute Data  -----
        # --------------------------
        print("Reading configs from {}:".format(str(root_path)))
        rtime = time()
        all_data = get_all_data(
            root_path, keep_pkl_keys, conf.get("multithreading", False), limit=5000
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
                dill.dump(cache, f)
            print("Done in {}s ({})".format(int(time() - t), sizeof(cache_path)))

    for plot in plots:
        func = all_plots[plot].run
        print_header(plot)
        try:
            # -------------------------------
            # -----  Run Plot Function  -----
            # -------------------------------
            func(data, plot_path, conf["compare"], **options[plot])

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
