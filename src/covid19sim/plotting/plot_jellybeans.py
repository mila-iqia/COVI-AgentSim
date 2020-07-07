from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.lines import Line2D
from collections import defaultdict
from covid19sim.plotting.utils.extract_data import get_all_rec_levels


def get_transformer_name(method_dict):
    for comparison in method_dict.values():
        for run in comparison.values():
            return Path(run["conf"]["TRANSFORMER_EXP_PATH"]).name


def run(data, path, comparison_key, use_wandb):
    """
    data:
        method:
            comparison_value:
                run:
                    conf: dict
                    pkl: dict

    Args:
        data ([type]): [description]
    """
    print("Preparing data...")
    intervention_day = None
    for mk, mv in data.items():
        if "unmitigated" in mk:
            continue
        for ck, cv in mv.items():
            for rk, rv in cv.items():
                intervention_day = rv["conf"]["INTERVENTION_DAY"]
                break

    assert intervention_day is not None

    colors = ["#007FA1", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]

    max_cols = 5

    data_rec_levels = {
        mk: {ck: get_all_rec_levels(data=cv) for ck, cv in mv.items()}
        for mk, mv in data.items()
    }

    tmp_data = defaultdict(dict)
    for mk, mrl in data_rec_levels.items():
        for ck, crl in mrl.items():
            tmp_data[ck][mk] = crl
    data_rec_levels = tmp_data

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markerfacecolor=color,
            markeredgecolor="k",
            markersize=15,
            label=f"Level {level - 1}",
        )
        for (level, color) in enumerate(colors)
    ]

    method_title = {
        "bdt1": "1st Order Binary Tracing",
        "bdt2": "2nd Order Binary Tracing",
        "heuristicv1": "Heuristic (v1)",
        "heuristicv2": "Heuristic (v2)",
        "transformer": "Transformer",
        "transformer-[1, 3, 5]": "Transformer-[1, 3, 5]",
        "transformer-[0, 1, 2]": "Transformer-[0, 1, 2]",
        "transformer-[0, 0, 0]": "Transformer-[0, 0, 0]",
        "linreg": "Linear Regression",
        "mlp": "MLP",
        "unmitigated": "Unmitigated",
        "oracle": "Oracle",
    }

    n_lines = np.math.ceil(len(data) / max_cols)
    n_cols = min((len(data), max_cols))

    for i, (comparison_value, comparison_dict) in enumerate(data_rec_levels.items()):
        fig = plt.figure(figsize=(5 * n_cols, 5 * n_lines), constrained_layout=True)
        gridspec = fig.add_gridspec(n_lines, n_cols)
        print(f"Plotting {comparison_key} {comparison_value} ...")
        for j, (method_name, method_risk_levels) in enumerate(comparison_dict.items()):
            if method_name == "unmitigated":
                continue
            col = j % max_cols
            row = j // max_cols
            title = method_title.get(
                method_name, method_name.replace("_", " ").capitalize()
            )

            transformer_name = None
            if method_name in {"transformer", "linreg", "mlp"}:
                transformer_name = get_transformer_name(data[method_name])
            if transformer_name is not None:
                title += " ({})".format(get_transformer_name(data[method_name]))

            ax = fig.add_subplot(gridspec[row, col])
            ax.stackplot(
                intervention_day + np.arange(method_risk_levels.shape[-1]) - 1,
                method_risk_levels.mean(0),
                colors=colors,
            )
            ax.axvspan(0, intervention_day - 1, fc="gray", alpha=0.2)
            ax.axvline(intervention_day - 1, c="k", ls="-.")
            ax.set_title(title, size=15)
            ax.tick_params(axis="both", which="major", labelsize=13)
            ax.yaxis.set_ticklabels(["0", "20", "40", "60", "80", "100"])
            ax.set_xlabel("Days", size=15)
            ax.margins(0, 0)
            if j == 0:
                ax.set_ylabel("% recommendation level", size=15)
                ax.legend(
                    handles=legend_handles,
                    loc="lower left",
                    framealpha=1.0,
                    fontsize=13,
                )
            else:
                ax.text(
                    intervention_day - 1.5,
                    0.05,
                    "Intervention",
                    size=15,
                    ha="right",
                    va="bottom",
                    rotation=90,
                )
        plt.suptitle("{} {}".format(comparison_key, comparison_value), size=17)
        save_path = path / "comparison-recommendation-levels-{}-{}.png".format(
            comparison_key, comparison_value
        )
        print("Saving Figure {} ...".format(save_path.name), end="", flush=True)
        plt.savefig(
            str(save_path), bbox_inches="tight",
        )
        if use_wandb:
            print("Uploading to Weights and Biases...", end="", flush=True)
            wandb.save(str(save_path))
        print("Done.")
