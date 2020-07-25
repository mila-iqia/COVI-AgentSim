import covid19sim
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from covid19sim.utils.constants import SECONDS_PER_DAY
from pathlib import Path


def run(data, path, comparison_key):

    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{method}_{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    for label, pkls in label2pkls:
        times = list()
        for data in pkls:
            data = data["infection_monitor"]
            for x in data:
                if x["from"]:
                    times.append((x["infection_timestamp"] - x["from_infection_timestamp"]).total_seconds() / SECONDS_PER_DAY)
        
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.hist(times, color="royalblue", bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        ax.set(title=f"Statistics of Generation Time {label}")
        ax.set_xlabel("Generation Time")
        ax.set_ylabel("Frequency")
        output_dir = os.path.join(path, "generation_times")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, f"GenerationTime_{label}.png")
        print(f"printing: {output_file}")
        plt.savefig(output_file)
