print("Loading imports...", end="", flush=True)
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import math
from collections import defaultdict

import covid19sim.plotting.plot_pareto_adoption as pareto_adoption
from covid19sim.plotting.utils.extract_data import get_all_paths

print("Ok.")
HYDRA_CONF_PATH = Path(__file__).parent.parent / "hydra-configs" / "plot"


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
    print("#" * 60)
    print()


def check_data(path, plots):
    assert path.exists()
    if "pareto_adoption" in plots:
        pass


def summarize_configs(all_paths):
    ignore_keys = {"outdir", "GIT_COMMIT_HASH", "DAILY_TARGET_REC_LEVEL_DIST"}
    total_pkls = sum(len(mv) for mv in all_paths.values())
    print(
        "Found {} methods and {} pkls:\n{}".format(
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
        "Varying parameters are:\n"
        + "    \n".join([f"{k}:{v}" for k, v in confs.items() if len(v) > 1])
    )


def get_model(conf, mapping):
    if conf["RISK_MODEL"] == "":
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return "unmitigated_norm"
        return "unmitigated"

    if conf["RISK_MODEL"] == "digital":
        if conf["TRACING_ORDER"] == 1:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "btd1_norm"
            return "bdt1"
        elif conf["TRACING_ORDER"] == 2:
            if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
                return "btd2_norm"
            return "bdt2"
        else:
            raise ValueError(
                "Unknown binary digital tracing order: {}".format(conf["TRACING_ORDER"])
            )

    if conf["RISK_MODEL"] == "transformer":
        # FIXME this won't work if the run used the inference server
        model = Path(conf["TRANSFORMER_EXP_PATH"]).name
        if model not in mapping:
            raise ValueError("Unknown transformer {}".format(model))
        if conf.get("DAILY_TARGET_REC_LEVEL_DIST", False):
            return conf["model_mapping"][model] + "_norm"
        return conf["model_mapping"][model]

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
            compare_value = plot_conf[key]
            model = get_model(sim_conf, plot_conf["model_mapping"])
            new_data[model][compare_value][rk] = rv
    return new_data


@hydra.main(config_path=str(HYDRA_CONF_PATH.resolve() / "config.yaml"), strict=False)
def main(conf):
    conf = OmegaConf.to_container(conf)

    all_plots = {"pareto_adoption": pareto_adoption}

    if "help" in conf:
        print("Available plotting options:")
        print("    * pareto_adoption")
        print("    * all")
        print('    * "[opt1, opt2, ...]" (<- note, lists are stringified)')
        print()
        print('Use exclude="[opt1, opt2, ...]" with `all` to run all plots but those')
        print()
        print("python main.py plot=pareto_adoption")
        print('python main.py plot=all exclude="[pareto_adoption]"')
        return

    path = Path(conf.get("path", ".")).resolve()

    plots = conf["plot"]
    if plots == "all":
        plots = list(all_plots.values())
    if not isinstance(plots, list):
        plots = [plots]

    check_data(path, plots)

    print("Reading configs from {}...".format(str(path)), end="", flush=True)
    all_paths = get_all_paths(path)
    print("Done.")
    summarize_configs(all_paths)
    data = map_conf_to_models(all_paths, conf)

    for plot in plots:
        func = all_plots[plot].run
        print_header(plot)
        func(data, conf["compare"])
        print_footer()


if __name__ == "__main__":
    main()
