import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


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
            if not any(data.get('covid_properties', [])):
                print("no covid!")
                return
            incubation.append(data["covid_properties"]["incubation_days"][1])
            infectiousness.append(data["covid_properties"]["infectiousness_onset_days"][1])
            recovery.append(data["covid_properties"]["recovery_days"][1])
            generation_time.append(data["generation_times"])

            u = sum(x[1] for x in data["daily_age_group_encounters"].values())
            v = sum(x for x in data["age_histogram"].values())
            daily_contact.append(float(u)/float(v))
            symptomatic = data["r_0"].get("symptomatic", {"infection_count": 0})["infection_count"]
            presymptomatic = data["r_0"].get("presymptomatic", {"infection_count": 0})["infection_count"]
            asymptomatic = data["r_0"].get("asymptomatic", {"infection_count": 0})["infection_count"]
            total = symptomatic + presymptomatic + asymptomatic
            total += sum([x for x in data["contacts"]["env_infection"].values()])
            if total:
                presymptomatic_transmission.append(presymptomatic/float(total))
                asymptomatic_transmission.append(asymptomatic/float(total))
            else:
                presymptomatic_transmission.append(0.)
                asymptomatic_transmission.append(0.)

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
        df = pd.DataFrame(result, columns=col_labels, index=row_labels)

        with open(os.path.join(path, "epi_table.md"), 'a+') as f:
            f.write("\n\n")
            f.write(label)
            f.write("\n")
            f.write(df.to_markdown())
