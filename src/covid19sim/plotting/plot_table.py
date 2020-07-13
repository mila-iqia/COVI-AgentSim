import covid19sim
import numpy as np
import scipy
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from covid19sim.utils.constants import SECONDS_PER_DAY

def run(data, path, comparison_key):

    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{method}_{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    for label, pkls in label2pkls:
        incubation = list()
        infectiousness = list()
        recovery = list()
        generation_time = list()
        daily_contact = list()
        presymptomatic_transmission = list()
        asymptomatic_transmission = list()

        for data in pkls:

            incubation.append(data["covid_properties"]["incubation_days"][1])
            infectiousness.append(data["covid_properties"]["infectiousness_onset_days"][1])
            recovery.append(data["covid_properties"]["recovery_days"][1])
            generation_time.append(data["generation_times"])

            u = sum(x[1] for x in data["daily_age_group_encounters"].values())
            v = sum(x for x in data["age_histogram"].values())
            daily_contact.append(float(u)/float(v))

            total = data["r_0"]["symptomatic"]["infection_count"] + data["r_0"]["presymptomatic"]["infection_count"] + data["r_0"]["asymptomatic"]["infection_count"]
            total += sum([x for x in data["contacts"]["env_infection"].values()])
            presymptomatic_transmission.append(data["r_0"]["presymptomatic"]["infection_count"]/float(total))
            asymptomatic_transmission.append(data["r_0"]["asymptomatic"]["infection_count"]/float(total))

        result = [
            [round(np.array(incubation).mean(), 2), round(np.array(incubation).std(), 2)],
            [round(np.array(infectiousness).mean(), 2), round(np.array(infectiousness).std(), 2)],
            [round(np.array(recovery).mean(), 2), round(np.array(recovery).std(), 2)],
            [round(np.array(generation_time).mean(), 2), round(np.array(generation_time).std(), 2)],
            [round(np.array(daily_contact).mean(), 2), round(np.array(daily_contact).std(), 2)],
            [round(np.array(presymptomatic_transmission).mean(), 2), round(np.array(presymptomatic_transmission).std(), 2)],
            [round(np.array(asymptomatic_transmission).mean(), 2), round(np.array(asymptomatic_transmission).std(), 2)]
        ]

        col_labels = ["Average", "Std Err"]
        row_labels = ["Incubation", "Infectiousness", "Recovery", "Generation Time", "Daily Contact", "Presymptomatic Transmission", "Asymptomatic Transmission"]

        table = plt.table(cellText=result, rowLabels=row_labels, colLabels=col_labels, loc="best", colWidths=[0.1, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.5)

        plt.title(f"{label}")
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
        for pos in ["right", "top", "bottom", "left"]:
            plt.gca().spines[pos].set_visible(False)

        output_file = os.path.join(path, f"Table_{label}.png")
        plt.savefig(output_file)
