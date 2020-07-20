import sys
import os
import datetime
import numpy as np
import pickle
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd 

path = "output/sim_v2_people-1000_days-150_init-0.002_uptake--1.0_seed-0_20200715-200646_797000/tracker_data_n_1000_seed_0_20200715-201949_.pkl"
path2 = "output/sim_v2_people-1000_days-150_init-0.002_uptake--1.0_seed-0_20200715-200646_797000/train_priors.pkl"
tracker_data = pickle.load(open(path, "rb"))
prior_data = pickle.load(open(path2, "rb"))

monitor = tracker_data['human_monitor']
test_monitor = tracker_data['test_monitor']
population = tracker_data['n_humans']
hospitalizations = [float(x)*100/population for x in prior_data['hospitalization_per_day']]
cases_per_day = [float(x)*100/population for x in tracker_data['cases_per_day']]

tests_per_day = Counter()
for test in test_monitor:
    date = test['test_time'].date()
    tests_per_day[date] += 100./population

dates = []
deaths = []

for k,v in monitor.items():
    dates.append(k)
    death = sum([x['dead'] for x in v])
    deaths.append(float(death)*100/population)
    # TODO: Add hospitalizations and case counts/positive tests

# PLOTS
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.plot(dates, deaths, label="Simulated Mortalities")
ax.plot(list(tests_per_day.keys()), list(tests_per_day.values()), label="Simulated Tests Per Day")
ax.plot(dates, hospitalizations[1:], label="Simulated Hospitalizations Per Day")
ax.plot(dates, cases_per_day[1:], label="Simulated Cases Per Day")
ax.legend()
plt.ylabel("Percentage of Population")
plt.xlabel("Date")
plt.xticks([x for i, x in enumerate(dates) if i % 10 == 0], rotation=45)
plt.title("Simulation Statistics")
plt.savefig("simulated.png")