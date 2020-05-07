import pandas as pd
import numpy as np

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

    def _get_posteriors(self, sr):
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
        #prior0 = sps.gamma(a=4).pdf(r_t_range)
        prior0 = np.ones_like(r_t_range)/len(r_t_range)
        for k in range(int(1.5/self.R_T_MAX*len(r_t_range)), int(2.5/self.R_T_MAX*len(r_t_range))):
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

    def compute(self, data):
        data = pd.Series(data, index=list(range(len(data))))
        posteriors, log_likelihood = self._get_posteriors(data)
        # Note that this takes a while to execute - it's not the most efficient algorithm
        hdis = self._highest_density_interval(posteriors, p=.9)
        most_likely = posteriors.idxmax().rename('ML')

        most_likely = np.array(most_likely)
        hdis = np.array(hdis)
        #hdis = 0

        return most_likely, hdis

    @staticmethod
    def plot(cases_per_day, true_R=None):
        plotrt = PlotRt(R_T_MAX=4, sigma=0.25)
        most_likely, hdis = plotrt.compute(cases_per_day)
        index = np.array(list(range(most_likely.shape[0])))
        plt.figure()
        plt.plot(index, np.ones(most_likely.shape[0]), label="Rt=1", color='green')
        plt.plot(index, most_likely, label="Rt Estimation", c='blue')
        #plt.scatter(index, most_likely, s=40, lw=.5, c='red', edgecolors='k', zorder=2)
        #plt.plot(index, hdis[:, 0], label="Rt Lower", color='cornflowerblue')
        #plt.plot(index, hdis[:, 1], label="Rt Upper", color='darkblue')
        lowfn = interp1d(index, hdis[:, 0], bounds_error=False, fill_value='extrapolate')
        highfn = interp1d(index, hdis[:, 1], bounds_error=False, fill_value='extrapolate')
        plt.fill_between(index, lowfn(index), highfn(index), color='k', alpha=.1, lw=0, zorder=3)
        if true_R != None:
            plt.plot(np.array(list(range(len(true_R)))), np.array(true_R), label="True Rt", color='red')
        plt.xlabel("Date")
        plt.ylabel("R")
        plt.title("Plot of Rt")
        plt.legend()
        plt.show()
        #plt.savefig('Rt', dpi=500)

# unit tests for the current class
if __name__ == '__main__':
    true_R = [2.0, 1.6666666666666667, 1.6666666666666667, 1.8, 1.8, 1.8333333333333333, 1.7142857142857142, 1.75]
    cases_per_day = [0, 0, 2, 2, 1, 1, 4, 2, 2, 6, 4, 6, 6, 7, 11, 7, 7, 18, 19, 11, 23, 34, 28, 29, 38, 60, 50, 56, 82, 81]
    PlotRt.plot(cases_per_day, true_R)
