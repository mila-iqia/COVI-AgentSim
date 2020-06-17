import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output

class PlotRt:
    def __init__(self, sigma=0.25, GAMMA=1.0/7.0, R_T_MAX=12):
        self.sigma = sigma
        self.GAMMA = GAMMA
        self.R_T_MAX = R_T_MAX

    def _highest_density_interval(self, pmf, p=.9, debug=False):
        # If we pass a DataFrame, just call this recursively on the columns
        if(isinstance(pmf, pd.DataFrame)):
            return pd.DataFrame([self._highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns)
        cumsum = np.cumsum(pmf.values)
        # N x N matrix of total probability mass for each low, high
        total_p = cumsum - cumsum[:, None]
        # Return all indices with total_p > p
        lows, highs = (total_p > p).nonzero()
        # Find the smallest range (highest density)
        best = (highs - lows).argmin()
        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]
        return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])

    def _get_posteriors(self, sr, r0_estimate=None):
        r_t_range = np.linspace(0, self.R_T_MAX, self.R_T_MAX*100+1)

        # (1) Calculate Lambda
        lam = sr[:-1].values * np.exp(self.GAMMA * (r_t_range[:, None] - 1)) + 1e-8

        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])

        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc=r_t_range, scale=self.sigma).pdf(r_t_range[:, None])

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior
        # prior0 = sps.gamma(a=4).pdf(r_t_range)
        prior0 = np.ones_like(r_t_range)/len(r_t_range)
        if r0_estimate:
            prior0[int(r0_estimate/self.R_T_MAX*len(r_t_range))] += 1e-8
        else:
            for k in range(int(1.0/self.R_T_MAX*len(r_t_range)), int(4.0/self.R_T_MAX*len(r_t_range))):
                prior0[k] = 10
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=r_t_range,
            columns=sr.index,
            data={sr.index[0]: prior0}
        )

        # We said we'd keep track of the sum of the log of the probability
        # of the data for maximum likelihood calculation.
        log_likelihood = 0.0

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

            #(5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]

            #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior

            #(5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            posteriors[current_day] = numerator/denominator

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        return posteriors, log_likelihood

    def _smooth_cases(self, cases):
        smoothed = list()
        for k in range(len(cases)):
            if k == 0:
                smoothed.append(math.ceil(2.0/3.0*cases[0]+1.0/3.0*cases[1]))
            elif k == len(cases) - 1:
                smoothed.append(math.ceil(2.0/3.0*cases[-1]+1.0/3.0*cases[-2]))
            else:
                smoothed.append(math.ceil(1.0/3.0*cases[k-1]+1.0/3.0*cases[k]+1.0/3.0*cases[k+1]))
        return smoothed

    def compute(self, data, r0_estimate=None, bounds=False):
        data = self._smooth_cases(data)
        data = pd.Series(data, index=list(range(len(data))))
        posteriors, log_likelihood = self._get_posteriors(data, r0_estimate)
        # Note that this takes a while to execute - it's not the most efficient algorithm
        hdis = False
        if bounds:
            hdis = self._highest_density_interval(posteriors, p=.5)
            hdis = np.array(hdis)

        most_likely = posteriors.idxmax().rename('ML')
        most_likely = np.array(most_likely)
        return most_likely, hdis

    @staticmethod
    def plot(ax, cases_per_day, color, marker, marker_size):
        plotrt = PlotRt(R_T_MAX=4, sigma=0.25)
        most_likely, hdis = plotrt.compute(cases_per_day)
        index = np.array(list(range(most_likely.shape[0])))
        ax.plot(index, most_likely, color=color, marker=marker, linestyle=":", alpha=0.5, ms=marker_size)
        lowfn = interp1d(index, hdis[:, 0], bounds_error=False, fill_value='extrapolate')
        highfn = interp1d(index, hdis[:, 1], bounds_error=False, fill_value='extrapolate')
        ax.fill_between(index, lowfn(index), highfn(index), color=color, alpha=.05, lw=0, zorder=3)
        return ax

# unit tests for the current class
if __name__ == '__main__':
    colormap = ['red', 'orange', 'blue', 'green', 'gray']
    end_day = 60
    R_marker = "P"
    R_marker_size = 10
    intervention_day = 10

    f, ax = plt.subplots(figsize=(20,10))

    line = ax.axhline(y=1.0, linestyle="-.", linewidth=3, color="green", alpha=0.5)
    ax.annotate("R = 1.0", xy=(intervention_day, 1.0), xytext=(intervention_day-10, 1.10), size=30, rotation="horizontal")

    a = [0, 0, 1, 3, 1, 1, 2, 2, 2, 3, 4, 11, 8, 9, 18, 19, 8, 22, 50, 35, 43, 71, 88, 55, 69, 150, 80, 66, 56, 46, 22, 10, 7, 9, 5, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 1, 3, 1, 1, 2, 2, 2, 3, 4, 11, 8, 9, 18, 19, 8, 16, 37, 24, 25, 42, 38, 38, 39, 72, 45, 43, 77, 61, 39, 51, 47, 22, 16, 16, 13, 13, 8, 5, 4, 4, 4, 1, 1, 3, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c = [0, 0, 1, 3, 1, 1, 2, 2, 2, 3, 4, 11, 8, 9, 18, 19, 8, 14, 10, 3, 10, 6, 9, 6, 3, 5, 3, 0, 2, 2, 2, 2, 2, 0, 4, 2, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1, 3, 0, 0, 0, 2, 1, 0, 0, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 3, 1, 0, 1, 3, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    all_data = [a, b, c]

    for i, data in enumerate(all_data):
        ax = PlotRt.plot(ax, data[:end_day], color=colormap[i], marker=R_marker, marker_size=R_marker_size)

    ax.set_ylabel('Rt', fontsize=30, rotation=0, labelpad=25)
    ax.set_ylim(0, 4)
    ax.tick_params(labelsize=25)
    plt.xlabel("Days since outbreak", fontsize=30)
    plt.savefig("Rt", dpi=500)
