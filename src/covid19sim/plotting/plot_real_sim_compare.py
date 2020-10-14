import os
import pickle
import yaml
import glob
import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

# Constants
quebec_population = 8485000
csv_path = "data/qc.csv"
sim_dir_path = "output/"
use_cache = False

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


from collections import defaultdict
results = defaultdict(list)
if not use_cache:
    for d in tqdm.tqdm(os.listdir(sim_dir_path)):
        source_path = os.path.join(sim_dir_path, d)
        config_path = os.path.join(sim_dir_path, d, "full_configuration.yaml")
        try:
            config = yaml.load(open(config_path, "rb"))
        except Exception as e:
            print(f"{e}, {config_path}")
            continue
        mob = config['GLOBAL_MOBILITY_SCALING_FACTOR']
        sick = config['init_fraction_sick']
        pop = config['n_people']
        days = config['simulation_days']
        name = f"mob_{mob}_sick_{sick}_pop_{pop}_days_{days}"

        sim_priors_path = os.path.join(source_path, "train_priors.pkl")
        sim_tracker_path = glob.glob(os.path.join(source_path, "*.pkl"))[0]

        sim_tracker_data = pickle.load(open(sim_tracker_path, "rb"))
        sim_prior_data = pickle.load(open(sim_priors_path, "rb"))
        sim_dates, sim_deaths, sim_tests, sim_cases = parse_tracker(sim_tracker_data)
        sim_hospitalizations = [float(x)*100/sim_tracker_data['n_humans'] for x in sim_prior_data['hospitalization_per_day']]
        results[name].append({"sim_dates": sim_dates, "sim_deaths": sim_deaths, "sim_tests": sim_tests, "sim_cases": sim_cases, "sim_hospitalizations": sim_hospitalizations})
    pickle.dump(results, open("cache_sim.pkl", "wb"))
    data = results
else:
    data = pickle.load(open("cache_sim.pkl", "rb"))

for k, v in data.items():
    print(k)
    sim_dates = v[0]['sim_dates']
    sim_deaths = np.array([x['sim_deaths'] for x in v])#.mean(axis=0)
    sim_cases = np.array([x['sim_cases'] for x in v])
    sim_hospitalizations = np.array([x['sim_hospitalizations'] for x in v])
   
    # Parse sim tests
    sim_tests = []
    min_length = min([len(x['sim_tests']) for x in v])
    for x in v:
        sim_test = x['sim_tests']
        sim_tests.append(list(sim_test.values())[:min_length])
    sim_tests = np.array(sim_tests)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    
    ax.errorbar(sim_dates, sim_deaths.mean(axis=0), yerr=sim_deaths.std(axis=0), label="Simulated Mortalities (per Day)")
    ax.errorbar(sim_dates, sim_hospitalizations.mean(axis=0)[1:], yerr=sim_hospitalizations.std(axis=0)[1:], label="Simulated Hospital Utilization (per Day)")
    ax.errorbar(sim_dates, sim_cases.mean(axis=0)[1:], yerr=sim_cases.std(axis=0)[1:], label="Simulated Cases (per Day)")
    ax.errorbar(sim_dates[:min_length], sim_tests.mean(axis=0), yerr=sim_tests.std(axis=0), label="Simulated Tests (per Day)")

    ax.legend()
    plt.ylabel("Percentage of Population")
    plt.xlabel("Date")
    plt.yticks(plt.yticks()[0], [str(round(x, 2)) + "%" for x in plt.yticks()[0]])
    plt.xticks([x for i, x in enumerate(sim_dates) if i % 10 == 0], rotation=45)
    plt.title("Quebec COVID Statistics")

    plt.savefig(f"{k}_sim_stats.png")


# Load data
qc_data = pd.read_csv(csv_path)


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

