import os, sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import scipy.stats as stats

# Constants
quebec_population = 8485000
csv_path = "path/to/csv"
sims_dir_path = "path/to/simulations/"
csv_path = "/mnt/c/Users/aksha/OneDrive - University of Waterloo/Desktop/mila/covi-simulator/src/covid19sim/plotting/qcdata.csv"
sims_dir_path = "/mnt/c/Users/aksha/OneDrive - University of Waterloo/Desktop/mila/covi-simulator/output/simz/"
if len(sys.argv) > 1:
    sims_dir_path = sys.argv[1]

# Load data
qc_data = pd.read_csv(csv_path)

# Utility Functions
def parse_tracker(sim_tracker_data):
    sim_pop = sim_tracker_data['n_humans']
    dates = []
    cumulative_deaths = []
    tests = Counter()

    # Get dates and deaths
    for k, v in sim_tracker_data['human_monitor'].items():
        dates.append(str(k))
        death = sum([x['dead'] for x in v])
        cumulative_deaths.append(death)

    daily_deaths_prop = []
    last_deaths = 0
    for idx, deaths in enumerate(cumulative_deaths):
        if idx == 0:
            daily_deaths_prop.append(0)
        else:
            deaths = (float(deaths) * 100 / sim_pop)
            daily_deaths_prop.append(deaths - last_deaths)
            last_deaths = deaths

    # Get tests
    for test in sim_tracker_data['test_monitor']:
        date = test['test_time'].date()
        tests[str(date)] += 100./sim_pop

    # Get cases
    cases = [float(x) * 100 / sim_pop for x in sim_tracker_data['cases_per_day']]

    return dates, daily_deaths_prop, tests, cases

all_sim_cases = []
all_sim_hospitalizations = []
all_sim_deaths = []

for sim in os.listdir(sims_dir_path):
    # Get the paths
    sim_path = os.path.join(sims_dir_path, sim)
    sim_priors_path = os.path.join(sim_path, "train_priors.pkl")
    sim_tracker_name = [str(f_name) for f_name in os.listdir(sim_path) if f_name.startswith("tracker_data")][0]
    sim_tracker_path = os.path.join(sim_path, sim_tracker_name)
    # Load the data
    sim_tracker_data = pickle.load(open(sim_tracker_path, "rb"))
    sim_prior_data = pickle.load(open(sim_priors_path, "rb"))
    # Parse data
    sim_dates, sim_deaths, sim_tests, sim_cases = parse_tracker(sim_tracker_data)
    sim_hospitalizations = [float(x)*100/sim_tracker_data['n_humans'] for x in sim_prior_data['hospital_usage_per_day']]
    # change key above in sim_prior_data['hospital_usage_per_day']

    all_sim_cases.append(sim_cases)
    all_sim_hospitalizations.append(sim_hospitalizations)
    all_sim_deaths.append(sim_deaths)

avg_sim_cases = np.array([sum(elem)/len(elem) for elem in zip(*all_sim_cases)])
avg_sim_hospitalizations = np.array([sum(elem)/len(elem) for elem in zip(*all_sim_hospitalizations)])
avg_sim_deaths = np.array([sum(elem)/len(elem) for elem in zip(*all_sim_deaths)])

# Plot Quebec Data
last_index = 63 # index of last day of Quebec data, use 63 for 30 days
fig, ax = plt.subplots(figsize=(7.5, 7.5))
real_dates = qc_data.loc[34:last_index, 'dates'].to_numpy()
real_cases = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:last_index, 'change_cases']]
# change key to 'total_hospitalizations' if needed
real_hospitalizations = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:last_index, 'total_hospitalizations']]
real_deaths = [100 * float(x if str(x) != "nan" else 0) / quebec_population for x in qc_data.loc[34:last_index,'change_fatalities']]

ax.plot(real_dates, real_cases, label="Diagnoses per day")
ax.plot(real_dates, real_hospitalizations, label="Hospital utilization per day")
ax.plot(real_dates, real_deaths, label="Mortalities per day")

ax.legend()
plt.ylabel("Percentage of Population")
plt.xlabel("Date")
plt.yticks(plt.yticks()[0], [str(round(x, 3)) + "%" for x in plt.yticks()[0]])
plt.xticks([x for i, x in enumerate(real_dates) if i % 10 == 0], rotation=45)
plt.title("Real Quebec COVID Statistics")
plt.savefig("qc_stats.png")

# Plot average across simulations
fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.set_ylim([0.,0.03])
avg_sim_cases = np.array(sim_cases[1:])*.1 # adjust for unknown cases

ax.errorbar(sim_dates, avg_sim_cases, yerr=avg_sim_cases.std(axis=0), label="Diagnoses per day")
# change caption below if using 'total_hospitalizations'
ax.errorbar(sim_dates, avg_sim_hospitalizations[1:], yerr=avg_sim_hospitalizations.std(axis=0), label="Hospital utilization per day")
ax.errorbar(sim_dates, avg_sim_deaths, yerr=avg_sim_deaths.std(axis=0), label="Mortalities per day")

ax.legend()
plt.ylabel("Percentage of Population")
plt.xlabel("Date")
plt.yticks(plt.yticks()[0], [str(round(x, 3)) + "%" for x in plt.yticks()[0]])
plt.xticks([x for i, x in enumerate(real_dates) if i % 10 == 0], rotation=45)
plt.title("Simulated Quebec COVID Statistics")
plt.savefig("sim_stats.png")
