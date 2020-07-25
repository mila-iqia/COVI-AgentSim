import sys
import os
import datetime
import numpy as np
import pickle
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd 

# REAL DATA
path = "COVID19Tracker.ca Data - QC.csv"
real_data = pd.read_csv(path)

real_dates = real_data.loc[34:,"dates"].to_numpy()
real_cases_per_day = [100 * float(x if str(x) != "nan" else 0) / 8485000 for x in real_data.loc[34:,"change_cases"]]
real_deaths = [100 * float(x if str(x) != "nan" else 0) / 8485000 for x in real_data.loc[34:,"total_fatalities"]]
real_tests_per_day = [100 * float(x if str(x) != "nan" else 0) / 8485000 for x in real_data.loc[34:,"change_tests"]]
real_hospitalizations_per_day = [100 * float(x if str(x) != "nan" else 0) / 8485000 for x in real_data.loc[34:, "change_hospitalizations"]]

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.plot(real_dates, real_cases_per_day, label="Real Cases Per Day")
ax.plot(real_dates, real_deaths, label="Real Deaths")
ax.plot(real_dates, real_tests_per_day, label="Real Tests Per Day")
ax.plot(real_dates, real_hospitalizations_per_day, label="Real Hospitalizations Per Day")

ax.legend()
plt.ylabel("Percentage of Population")
plt.xlabel("Date")
plt.xticks([x for i, x in enumerate(real_dates) if i % 10 == 0], rotation=45)
plt.title("Quebec COVID Statistics")
plt.savefig("real_data.png")

