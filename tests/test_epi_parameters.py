import datetime
import os
from scipy.optimize import curve_fit
from scipy.stats import lognorm, norm, gamma
import yaml
from omegaconf import OmegaConf
import numpy as np

import matplotlib.pyplot as plt

import covid19sim
from covid19sim.base import City, Env
from covid19sim.simulator import Human
from covid19sim.utils import parse_configuration


def load_config():
    HYDRA_PATH = os.path.dirname(covid19sim.__file__) + "/hydra-configs/simulation/"
    assert os.path.isdir(HYDRA_PATH)

    config_path = os.path.join(HYDRA_PATH, "config.yaml")

    with open(config_path, "r") as fd:
        defaults = yaml.safe_load(fd)["defaults"]

    default_confs = [
        OmegaConf.load(os.path.join(HYDRA_PATH, d + ".yaml"))
        for d in defaults
    ]

    conf = OmegaConf.merge(*default_confs)
    return parse_configuration(conf)

def test_incubation_days():
    """
    Intialize `Human`s and compute their covid properties.
    Test whether incubation days follow a lognormal distribution with mean 5 days and scale 2.5 days.
    Refer Table 2 (Appendix) in https://www.acpjournals.org/doi/10.7326/M20-0504 for parameters of lognormal fit
    Reference values: mu= 1.621 (1.504 - 1.755) sigma=0.418 (0.271 - 0.542)
    """
    conf = load_config()

    def lognormal_func(x, mu, sigma):
        return lognorm.pdf(x, s=sigma, loc=0, scale=np.exp(mu))

    def normal_func(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)

    def gamma_func(x, shape, scale):
        return gamma.pdf(x, a=shape, scale=scale)

    N = 20
    rng = np.random.RandomState(42)
    fitted_incubation_params = []
    fitted_infectiousness_onset_params = []
    fitted_recovery_params = []
    # using matplotlib as a way to obtain density. TODO: use numpy
    fig, ax = plt.subplots()
    for i in range(N):
        n_people = rng.randint(500,1000)
        init_percent_sick = rng.uniform(0.01, 0.05)
        start_time = datetime.datetime(2020, 2, 28, 0, 0)

        env = Env(start_time)
        city_x_range = (0, 1000)
        city_y_range = (0, 1000)
        city = City(
            env,
            n_people,
            init_percent_sick,
            rng,
            city_x_range,
            city_y_range,
            Human,
            conf,
        )

        incubation_data, infectiousness_onset_data, recovery_data = [], [], []
        for human in city.humans:
            human.compute_covid_properties()
            assert human.incubation_days >= 0, "negative incubation days"
            assert human.infectiousness_onset_days >= 0, "negative infectiousness onset days"
            assert human.recovery_days >= 0, "negative recovery days"
            incubation_data.append(human.incubation_days)
            infectiousness_onset_data.append(human.infectiousness_onset_days)
            recovery_data.append(human.recovery_days)

        print(f"minimum incubation days: {min(incubation_data)}")
        # convert into pmf
        ydata = np.array(incubation_data)
        pmf, xdata, _ = ax.hist(ydata, density=True)
        xdata = np.array([(xdata[i] + xdata[i+1])/2 for i in range(0,xdata.shape[0]-1)])
        popt, pcov = curve_fit(gamma_func, xdata, pmf)
        fitted_incubation_params.append(popt)

        # convert into pmf
        ydata = np.array(infectiousness_onset_data)
        pmf, xdata, _ = ax.hist(ydata, density=True)
        xdata = np.array([(xdata[i] + xdata[i+1])/2 for i in range(0,xdata.shape[0]-1)])
        popt, pcov = curve_fit(gamma_func, xdata, pmf)
        fitted_infectiousness_onset_params.append(popt)

        # convert into pmf
        ydata = np.array(recovery_data)
        pmf, xdata, _ = ax.hist(ydata, density=True)
        xdata = np.array([(xdata[i] + xdata[i+1])/2 for i in range(0,xdata.shape[0]-1)])
        popt, pcov = curve_fit(normal_func, xdata, pmf, bounds=(14,30))
        fitted_recovery_params.append(popt)

    param_names = ["incubation days", "infectiousness onset days", "recovery days"]
    for idx, fitted_params in enumerate([fitted_incubation_params, fitted_infectiousness_onset_params, fitted_recovery_params]):
        all_params = np.array(fitted_params)

        # shape
        avg_mu, std_mu = all_params[:,0].mean(), all_params[:,0].std()
        ci_mu = norm.interval(0.95, loc=avg_mu, scale=std_mu)

        # scale
        avg_sigma, std_sigma = all_params[:,1].mean(), all_params[:,1].std()
        ci_sigma = norm.interval(0.95, loc=avg_sigma, scale=std_sigma)

        if param_names[idx] == "incubation days":
            print(f"**** Fitted Gamma distribution over {N} runs ... 95% CI ****")
            print(f"{param_names[idx]}")
            print(f"shape: {avg_mu: 3.3f} ({ci_mu[0]: 3.3f} - {ci_mu[1]: 3.3f}) refernce value: 5.807 (3.585 - 13.865)")
            print(f"scale: {avg_sigma: 3.3f} ({ci_sigma[0]: 3.3f} - {ci_sigma[1]: 3.3f}) refernce value: 0.948 (0.368 - 1.696)")
            assert 3.585 <= avg_mu <= 13.865, "not a fitted gamma distribution"

        elif param_names[idx] == "infectiousness onset days":
            print(f"**** Fitted Gamma distribution over {N} runs ... 95% CI ****")
            print(f"{param_names[idx]}")
            print(f"shape: {avg_mu: 3.3f} ({ci_mu[0]: 3.3f} - {ci_mu[1]: 3.3f}) refernce value: mean is 5.807-2.3 = 3.507 days (refer paramters in core.yaml)")
            print(f"scale: {avg_sigma: 3.3f} ({ci_sigma[0]: 3.3f} - {ci_sigma[1]: 3.3f}) refernce value: no-source")

        elif param_names[idx] == "recovery days":
            print(f"**** Fitted Normal distribution over {N} runs ... 95% CI ****")
            print(f"{param_names[idx]}")
            print(f"mu: {avg_mu: 3.3f} ({ci_mu[0]: 3.3f} - {ci_mu[1]: 3.3f}) refernce value: mean is 14 + 5.807 = 19.807 days (refer paramters in core.yaml)")
            print(f"sigma: {avg_sigma: 3.3f} ({ci_sigma[0]: 3.3f} - {ci_sigma[1]: 3.3f}) refernce value: no-source")


if __name__ == "__main__":
    test_incubation_days()
