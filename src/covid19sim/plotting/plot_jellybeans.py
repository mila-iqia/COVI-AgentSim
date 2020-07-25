import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from collections import defaultdict
from covid19sim.plotting.utils import get_all_rec_levels, get_title


def get_transformer_name(method_dict):
    for comparison in method_dict.values():
        for run in comparison.values():
            return Path(run["conf"]["TRANSFORMER_EXP_PATH"]).name


def run(data, path, comparison_key):
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
    # TODO: fix this loop in the context of different intervention days
    for mk, mv in data.items():
        if "unmitigated" in mk:
            continue
        for ck, cv in mv.items():
            for rk, rv in cv.items():
                intervention_day = rv["conf"]["INTERVENTION_DAY"]
                break

    assert intervention_day is not None

    colors = ["#007FA1", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]

    max_cols = 2

    data_rec_levels = {
        mk: {
            ck: get_all_rec_levels(data=cv, normalized="_norm" in mk)
            for ck, cv in mv.items()
        }
        for mk, mv in data.items()
    }

    tmp_data = defaultdict(dict)
    for mk, mrl in data_rec_levels.items():
        if "unmitigated" not in mk and "no_intervention" not in mk:
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

    n_lines = np.math.ceil(len(data) / max_cols)
    n_cols = min((len(data), max_cols))

    for i, (comparison_value, comparison_dict) in enumerate(data_rec_levels.items()):
        fig = plt.figure(figsize=(8 * n_cols, 8 * n_lines), constrained_layout=True,)
        gridspec = fig.add_gridspec(n_lines, n_cols)
        print(f"Plotting {comparison_key} {comparison_value}...")

        method_names = sorted(comparison_dict.keys())
        for j, method_name in enumerate(method_names):
            method_risk_levels = comparison_dict[method_name]

            col = j % max_cols
            row = j // max_cols
            title = get_title(method_name)

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
            ax.set_title(title, size=35)
            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.yaxis.set_ticklabels(["0", "20", "40", "60", "80", "100"])
            ax.set_xlabel("Days", size=30)
            ax.margins(0, 0)
            if j == 0:
                ax.set_ylabel("% recommendation level", size=33)
                ax.legend(
                    handles=legend_handles,
                    loc="lower left",
                    framealpha=1.0,
                    fontsize=23,
                )
            else:
                ax.text(
                    intervention_day - 1.5,
                    0.05,
                    "Intervention",
                    size=25,
                    ha="right",
                    va="bottom",
                    rotation=90,
                )
        plt.suptitle("{} {}".format(comparison_key, comparison_value), size=47)
        save_path = (
            path
            / "jellybeans/comparison-recommendation-levels-{}-{}.png".format(
                comparison_key, comparison_value
            )
        )
        os.makedirs(save_path.parent, exist_ok=True)
        print("Saving Figure {}...".format(save_path.name), end="", flush=True)
        plt.savefig(
            str(save_path), bbox_inches="tight",
        )
        print("Done.")
