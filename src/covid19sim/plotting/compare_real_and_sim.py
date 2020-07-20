import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

# Constants
quebec_population = 8485000
csv_path = "COVID19Tracker.ca Data - QC.csv"
sim_dir_path = "results/mob_pop_sick/10k_005/sim_v2_people-10000_days-30_init-0.005_uptake--1.0_seed-5002_20200716-181338_103092/"
#sim_dir_path = "output/sim_v2_people-1000_days-150_init-0.004_uptake--1.0_seed-0_20200716-101640_592894/"
sim_priors_path = os.path.join(sim_dir_path, "train_priors.pkl")
# sim_tracker_path = os.path.join(sim_dir_path, "tracker_data_n_100_seed_0_20200716-101538_.pkl")
sim_tracker_path = os.path.join(sim_dir_path, "tracker_data_n_10000_seed_5002_20200716-185829_.pkl")



# Load data
qc_data = pd.read_csv(csv_path)
sim_tracker_data = pickle.load(open(sim_tracker_path, "rb"))
sim_prior_data = pickle.load(open(sim_priors_path, "rb"))

# Utility Functions

def parse_tracker(sim_tracker_data):
    sim_pop = sim_tracker_data['n_humans']
    dates = []
    deaths = []
    tests = Counter()

    # Get dates and deaths
    for k, v in sim_tracker_data['human_monitor'].items():
        dates.append(str(k))
        death = sum([x['dead'] for x in v])
        deaths.append(float(death) * 100 / sim_pop)
        # TODO: Add hospitalizations and case counts/positive tests

    # Get tests
    for test in sim_tracker_data['test_monitor']:
        date = test['test_time'].date()
        tests[str(date)] += 100./sim_pop

    # Get cases
    cases = [float(x) * 100 / sim_pop for x in sim_tracker_data['cases_per_day']]

    return dates, deaths, tests, cases

# Parse data
sim_dates, sim_deaths, sim_tests, sim_cases = parse_tracker(sim_tracker_data)
sim_hospitalizations = [float(x)*100/sim_tracker_data['n_humans'] for x in sim_prior_data['hospitalization_per_day']]

# Plot cases
fig, ax = plt.subplots(figsize=(7.5, 7.5))
real_dates = qc_data.loc[34:, "dates"].to_numpy()
real_cases = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:, "change_cases"]]
real_hospitalizations = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:, "total_hospitalizations"]]
real_deaths = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:,"change_fatalities"]]
real_tests = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:, "change_tests"]]

ax.plot(real_dates, real_cases, label="QC Cases (per Day)")
ax.plot(real_dates, real_hospitalizations, label="QC Hospital Utilization (by Day)")
ax.plot(real_dates, real_deaths, label="QC Mortalities (per Day)")
ax.plot(real_dates, real_tests, label="QC Tests (per Day)")

ax.legend()
plt.ylabel("Percentage of Population")
plt.xlabel("Date")
plt.yticks(plt.yticks()[0], [str(round(x, 2)) + "%" for x in plt.yticks()[0]])
plt.xticks([x for i, x in enumerate(real_dates) if i % 10 == 0], rotation=45)
plt.title("Quebec COVID Statistics")
plt.savefig("qc_stats.png")




# Plot deaths and hospitalizations
fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.plot(sim_dates, sim_deaths, label="Simulated Mortalities (per Day)")
ax.plot(sim_dates, sim_hospitalizations[1:], label="Simulated Hospital Utilization (per Day)")
ax.plot(sim_dates, sim_cases[1:], label="Simulated Cases (per Day)")
ax.plot(list(sim_tests.keys()), list(sim_tests.values()), label="Simulated Tests (per Day)")

ax.legend()
plt.ylabel("Percentage of Population")
plt.xlabel("Date")
plt.yticks(plt.yticks()[0], [str(round(x, 2)) + "%" for x in plt.yticks()[0]])
plt.xticks([x for i, x in enumerate(real_dates) if i % 10 == 0], rotation=45)
plt.title("Quebec COVID Statistics")
plt.savefig("sim_stats.png")

