import numpy as np
from matplotlib import pyplot as plt
from utils import sample_viral_load

"""Samples the viral_load function 100 times and output a chart of the viral load over num_days"""
NUM_DAYS = 9
viral_loads = []
x = np.linspace(1, NUM_DAYS, NUM_DAYS)

for i in range(100):
    vals = _sample_viral_load().pdf(x)
    viral_loads.append(vals)
viral_loads = np.array(viral_loads)

print(f"viral_loads means: {viral_loads.mean(axis=0)}")
print(f"viral_loads stds: {viral_loads.std(axis=0)}")

fig, ax = plt.subplots(1, 1)
ax.errorbar(x, viral_loads.mean(axis=0), yerr=viral_loads.std(axis=0), lw=5, alpha=0.6, label='gamma pdf')
plt.xlabel("Days since infection")
plt.ylabel("Viral load")
plt.title("Viral Load (Noisy Gamma Model)")
plt.savefig("output/viral_load.jpg")
