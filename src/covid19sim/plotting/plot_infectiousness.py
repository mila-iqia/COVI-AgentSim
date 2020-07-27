import numpy as np
import matplotlib.pyplot as plt

incubation_days_np = np.load(open("incubation_days.npy", "rb"))
infectiousness_onset_days_np = np.load(open("infectiousness_onset_days.npy", "rb"))
infectiousness_onset_days_wrt_incubation_np = np.load(open("infectiousness_onset_days_wrt_incubation.npy", "rb"))

plt.title("infectiousness days")
plt.hist(infectiousness_onset_days_np, bins=range(0, 20, 1))
plt.savefig("infectiousness_onset_days.png")
plt.clf()

plt.title("incubation days")
plt.hist(incubation_days_np, bins=range(0, 20, 1))
plt.savefig("incubation_days.png")
plt.clf()

plt.title("infectiousness_onset_days_wrt_incubation_np")
plt.hist(infectiousness_onset_days_wrt_incubation_np, bins=range(0, 20, 1))
plt.savefig("infectiousness_onset_days_wrt_incubation.png")
plt.clf()
