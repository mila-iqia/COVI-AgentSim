import hydra
from omegaconf import OmegaConf
from pathlib import Path
import math
from collections import defaultdict

import covid19sim.plotting.plot_pareto_adoption as pareto_adoption
from covid19sim.plotting.utils.extract_data import get_all_paths

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
    print("Found {} methods and {} pkls:\n{}".format(
        len(all_paths),
        total_pkls,
        "\n    ".join(["{} ({})".format(Path(mk).name, len(mv)) for mk, mv in all_paths.items()])
    ))
    confs = defaultdict(set)
    for mk, mv in all_paths.items():
        for rk, rv in mv.items():
            for ck, cv in rv["conf"].items():
                if ck in ignore_keys:
                    continue
                confs[ck].add(str(cv))
    print("Varying parameters are:\n" + "    \n".join([f"{k}:{v}" for k, v in confs.items() if len(v) > 1]))

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
    
    import pdb; pdb.set_trace()
    
    for plot in plots:
        func = all_plots[plot].run
        print_header(plot)
        func(all_paths)
        print_footer()


if __name__ == "__main__":
    main()
