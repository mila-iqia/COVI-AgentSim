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
from covid19sim.human import Human
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


def test_human_compute_covid_properties():
    """
    Test the covid properties of the class Human over a population for 3 ages
    """
    conf = load_config()

    n_people = 10000
    init_percent_sick = 0
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)

    env = Env(start_time)

    city = City(
        env,
        1,  # This test directly calls Human.compute_covid_properties() on a Human
        init_percent_sick,
        np.random.RandomState(42),
        city_x_range,
        city_y_range,
        Human,
        conf,
    )

    def _get_human_covid_properties(human):
        human.compute_covid_properties()

        assert human.infectiousness_onset_days >= 1.0
        assert human.viral_load_peak_start >= 0.5 - 0.00001
        assert human.incubation_days >= 2.0

        assert human.infectiousness_onset_days < human.incubation_days

        # viral_load_peak_start, viral_load_plateau_start and viral_load_plateau_
        # end are relative to infectiousness_onset_days
        assert human.infectiousness_onset_days < human.viral_load_peak_start + human.infectiousness_onset_days
        assert human.viral_load_peak_start + human.infectiousness_onset_days < \
               human.incubation_days
        assert human.incubation_days < human.viral_load_plateau_start + human.infectiousness_onset_days
        assert human.viral_load_plateau_start < human.viral_load_plateau_end
        assert human.viral_load_plateau_end + human.infectiousness_onset_days < \
               human.recovery_days

        # &infectiousness-onset [He 2020 https://www.nature.com/articles/s41591-020-0869-5#ref-CR1]
        # infectiousness started from 2.3 days (95% CI, 0.8–3.0 days) before symptom
        # onset and peaked at 0.7 days (95% CI, −0.2–2.0 days) before symptom onset (Fig. 1c).
        assert human.incubation_days - human.infectiousness_onset_days >= 0.5
        assert human.incubation_days - human.infectiousness_onset_days <= 3.3

        # &infectiousness-onset [He 2020 https://www.nature.com/articles/s41591-020-0869-5#ref-CR1]
        # infectiousness peaked at 0.7 days (95% CI, −0.2–2.0 days) before symptom onset (Fig. 1c).
        try:
            assert human.incubation_days - \
                   (human.viral_load_peak_start + human.infectiousness_onset_days) >= 0.01
        except AssertionError:
            # If the assert above fails, it can only be when we forced viral_load_peak_start
            # to 0.5 day after infectiousness_onset_days
            assert abs(human.viral_load_peak_start - 0.5) <= 0.00001
        assert human.incubation_days - \
               (human.viral_load_peak_start + human.infectiousness_onset_days) <= 2.2

        # Avg plateau duration
        # infered from https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf (Figure 1 & 4).
        # 8 is infered from Figure 4 by eye-balling.
        assert human.viral_load_plateau_end - human.viral_load_plateau_start >= 3.0
        assert human.viral_load_plateau_end - human.viral_load_plateau_start <= 9.0

        assert human.viral_load_peak_height >= conf['MIN_VIRAL_LOAD_PEAK_HEIGHT']
        assert human.viral_load_peak_height <= conf['MAX_VIRAL_LOAD_PEAK_HEIGHT']

        assert human.viral_load_plateau_height <= human.viral_load_peak_height

        # peak_plateau_slope must transit from viral_load_peak_height to
        # viral_load_plateau_height
        assert (human.viral_load_peak_height -
                human.peak_plateau_slope * (human.viral_load_plateau_start -
                                            human.viral_load_peak_start)) - \
               human.viral_load_plateau_height < 0.00001

        # peak_plateau_slope must transit from viral_load_plateau_height to 0.0
        assert human.viral_load_plateau_height - \
               human.plateau_end_recovery_slope * (human.recovery_days -
                                                   (human.viral_load_plateau_end +
                                                    human.infectiousness_onset_days)) < 0.00001

        return [human.infectiousness_onset_days, human.viral_load_peak_start,
                human.incubation_days, human.viral_load_plateau_start,
                human.viral_load_plateau_end, human.recovery_days,
                human.viral_load_peak_height, human.viral_load_plateau_height,
                human.peak_plateau_slope, human.plateau_end_recovery_slope]

    human = city.humans[0]
    # Reset the rng
    human.rng = np.random.RandomState(42)
    # force is_asymptomatic to True since we are not testing the symptoms
    human.is_asymptomatic = True
    # force the age to a median
    human.age = 40
    covid_properties_samples = [_get_human_covid_properties(human)
                                for _ in range(n_people)]

    covid_properties_samples_mean = covid_properties_samples[0]
    for sample in covid_properties_samples[1:]:
        for i in range(len(covid_properties_samples_mean)):
            covid_properties_samples_mean[i] += sample[i]

    for i in range(len(covid_properties_samples_mean)):
        covid_properties_samples_mean[i] /= n_people

    infectiousness_onset_days_mean, viral_load_peak_start_mean, \
        incubation_days_mean, viral_load_plateau_start_mean, \
        viral_load_plateau_end_mean, recovery_days_mean, \
        viral_load_peak_height_mean, viral_load_plateau_height_mean, \
        peak_plateau_slope_mean, plateau_end_recovery_slope_mean = covid_properties_samples_mean

    # infectiousness_onset_days
    # &infectiousness-onset [He 2020 https://www.nature.com/articles/s41591-020-0869-5#ref-CR1]
    # infectiousness started from 2.3 days (95% CI, 0.8–3.0 days) before symptom
    # onset and peaked at 0.7 days (95% CI, −0.2–2.0 days) before symptom onset (Fig. 1c).
    # TODO: infectiousness_onset_days has a minimum of 1 which affects this mean. Validate this assert
    assert abs(infectiousness_onset_days_mean - 2.3) < 1.5, \
        f"The average of infectiousness_onset_days should be about {2.3}"

    # viral_load_peak_start
    # &infectiousness-onset [He 2020 https://www.nature.com/articles/s41591-020-0869-5#ref-CR1]
    # infectiousness peaked at 0.7 days (95% CI, −0.2–2.0 days) before symptom onset (Fig. 1c).
    assert abs(incubation_days_mean -
               (viral_load_peak_start_mean + infectiousness_onset_days_mean) - 0.7) < 0.5, \
        f"The average of viral_load_peak_start should be about {0.7}"

    # incubation_days
    # INCUBATION PERIOD
    # Refer Table 2 (Appendix) in https://www.acpjournals.org/doi/10.7326/M20-0504 for parameters of lognormal fit
    assert abs(incubation_days_mean - 5.807) < 0.5, \
        f"The average of infectiousness_onset_days should be about {5.807} days"

    # viral_load_plateau_start_mean, viral_load_plateau_end_mean
    # Avg plateau duration
    # infered from https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf (Figure 1 & 4).
    # 8 is infered from Figure 4 by eye-balling.
    assert abs(viral_load_plateau_end_mean - viral_load_plateau_start_mean) - 4.5 < 0.5, \
        f"The average of the plateau duration should be about {4.5} days"

    # (no-source) 14 is loosely defined.
    assert abs(recovery_days_mean - incubation_days_mean) - 14 < 0.5, \
        f"The average of the recovery time  should be about {14} days"

    # Test with a young and senior ages
    for age in (20, 75):
        human.age = age
        for _ in range(n_people):
            # Assert the covid properties
            _get_human_covid_properties(human)


def test_human_viral_load_for_day():
    """
    Test the sample over the viral load curve
    """
    conf = load_config()

    init_percent_sick = 0
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)

    env = Env(start_time)

    city = City(
        env,
        1,  # This test force the call Human.compute_covid_properties()
        init_percent_sick,
        np.random.RandomState(42),
        city_x_range,
        city_y_range,
        Human,
        conf,
    )

    human = city.humans[0]
    # force is_asymptomatic to True since we are not testing the symptoms
    human.is_asymptomatic = True
    # force the age to a median
    human.age = 40
    # Set infection date
    now = env.timestamp
    human.infection_timestamp = now

    # Curve key points in days wrt infection timestamp
    # Set plausible covid properties to make the computations easy to validate
    infectiousness_onset_days = 2.5
    viral_load_peak_start = 4.5
    incubation_days = 5
    viral_load_plateau_start = 5.5
    viral_load_plateau_end = 5.5 + 4.5
    recovery_days = 5 + 15

    human.infectiousness_onset_days = infectiousness_onset_days
    # viral_load_peak_start, viral_load_plateau_start and viral_load_plateau_
    # end are relative to infectiousness_onset_days
    human.viral_load_peak_start = viral_load_peak_start - infectiousness_onset_days
    human.incubation_days = incubation_days
    human.viral_load_plateau_start = viral_load_plateau_start - infectiousness_onset_days
    human.viral_load_plateau_end = viral_load_plateau_end - infectiousness_onset_days
    human.recovery_days = recovery_days

    human.viral_load_peak_height = 1.0
    human.viral_load_plateau_height = 0.75
    human.peak_plateau_slope = 0.25 / (viral_load_plateau_start - viral_load_peak_start)
    human.plateau_end_recovery_slope = 0.75 / (recovery_days - viral_load_plateau_end)

    assert human.viral_load_for_day(now) == 0.0
    # Between infection_timestamp and infectiousness_onset_days
    assert human.viral_load_for_day(now +
                                    datetime.timedelta(days=infectiousness_onset_days / 2)) == 0.0
    assert human.viral_load_for_day(now + datetime.timedelta(days=infectiousness_onset_days)) == 0.0
    # Between infectiousness_onset_days and viral_load_peak_start
    assert human.viral_load_for_day(now +
                                    datetime.timedelta(days=infectiousness_onset_days +
                                                            (viral_load_peak_start - infectiousness_onset_days) /
                                                            2)) == 1.0 / 2
    assert human.viral_load_for_day(now + datetime.timedelta(days=viral_load_peak_start)) == 1.0
    assert human.viral_load_for_day(now + datetime.timedelta(days=incubation_days)) == 0.75 + 0.25 / 2
    assert human.viral_load_for_day(now + datetime.timedelta(days=viral_load_plateau_start)) == 0.75
    # Between viral_load_plateau_start and viral_load_plateau_end
    assert human.viral_load_for_day(now +
                                    datetime.timedelta(days=viral_load_plateau_start +
                                                            (viral_load_plateau_end - viral_load_plateau_start) /
                                                            2)) == 0.75
    assert human.viral_load_for_day(now + datetime.timedelta(days=viral_load_plateau_end)) == 0.75
    assert human.viral_load_for_day(now +
                                    datetime.timedelta(days=viral_load_plateau_end +
                                                            (recovery_days - viral_load_plateau_end) /
                                                            2)) == 0.75 / 2
    assert human.viral_load_for_day(now + datetime.timedelta(days=recovery_days)) == 0.0


class EnvMock():
        """
        Custom simpy.Environment
        """

        def __init__(self, initial_timestamp):
            """
            Args:
                initial_timestamp (datetime.datetime): The environment's initial timestamp
            """
            self.initial_timestamp = datetime.datetime.combine(initial_timestamp.date(),
                                                               datetime.time())
            self.ts_initial = int(self.initial_timestamp.timestamp())
            self.now = self.ts_initial

        @property
        def timestamp(self):
            """
            Returns:
                datetime.datetime: Current date.
            """
            #
            ##
            ## The following is preferable, but not equivalent to the initial
            ## version, because timedelta ignores Daylight Saving Time.
            ##
            #
            # return datetime.datetime.fromtimestamp(int(self.now))
            #
            return self.initial_timestamp + datetime.timedelta(
                seconds=self.now - self.ts_initial)

        def minutes(self):
            """
            Returns:
                int: Current timestamp minute
            """
            return self.timestamp.minute

        def hour_of_day(self):
            """
            Returns:
                int: Current timestamp hour
            """
            return self.timestamp.hour

        def day_of_week(self):
            """
            Returns:
                int: Current timestamp day of the week
            """
            return self.timestamp.weekday()

        def is_weekend(self):
            """
            Returns:
                bool: Current timestamp day is a weekend day
            """
            return self.day_of_week() >= 5

        def time_of_day(self):
            """
            Time of day in iso format
            datetime(2020, 2, 28, 0, 0) => '2020-02-28T00:00:00'

            Returns:
                str: iso string representing current timestamp
            """
            return self.timestamp.isoformat()


def test_human_cold_symptoms():
    conf = load_config()

    # Test cold symptoms
    conf["P_COLD_TODAY"] = 1.0
    conf["P_FLU_TODAY"] = 0.0
    conf["P_HAS_ALLERGIES_TODAY"] = 0.0
    init_percent_sick = 0

    n_people = 10000
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)

    env = EnvMock(start_time)

    # Init humans
    city = City(
        env,
        n_people,
        init_percent_sick,
        np.random.RandomState(42),
        city_x_range,
        city_y_range,
        Human,
        conf
    )

    for day in range(10):
        env.now += SECONDS_PER_DAY
        for human in city.humans:
            human.has_allergies = False
            human.catch_other_disease_at_random()
            human.update_symptoms()
            if day < len(human.cold_progression):
                assert set(human.all_symptoms) == set(human.cold_progression[day]), \
                    f"Human symptoms should be those of cold"


def test_human_flu_symptoms():
    conf = load_config()

    # Test flu symptoms
    conf["P_COLD_TODAY"] = 0.0
    conf["P_FLU_TODAY"] = 1.0
    conf["P_HAS_ALLERGIES_TODAY"] = 0.0
    init_percent_sick = 0

    n_people = 10000
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)

    env = EnvMock(start_time)

    # Init humans
    city = City(
        env,
        n_people,
        init_percent_sick,
        np.random.RandomState(42),
        city_x_range,
        city_y_range,
        Human,
        conf
    )

    for day in range(10):
        env.now += SECONDS_PER_DAY
        for human in city.humans:
            human.has_allergies = False
            human.catch_other_disease_at_random()
            human.update_symptoms()
            if day < len(human.flu_progression):
                assert set(human.all_symptoms) == set(human.flu_progression[day]), \
                    f"Human symptoms should be those of flu"


def test_human_allergies_symptoms():
    conf = load_config()

    # Test allergies symptoms
    conf["P_COLD_TODAY"] = 0.0
    conf["P_FLU_TODAY"] = 0.0
    conf["P_HAS_ALLERGIES_TODAY"] = 1.0
    init_percent_sick = 0

    n_people = 10000
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    city_x_range = (0, 1000)
    city_y_range = (0, 1000)

    env = EnvMock(start_time)

    # Init humans
    city = City(
        env,
        n_people,
        init_percent_sick,
        np.random.RandomState(42),
        city_x_range,
        city_y_range,
        Human,
        conf
    )

    for day in range(10):
        env.now += SECONDS_PER_DAY
        for human in city.humans:
            human.has_allergies = True
            human.catch_other_disease_at_random()
            human.update_symptoms()
            if day < len(human.allergy_progression):
                assert set(human.all_symptoms) == set(human.allergy_progression[day]), \
                    f"Human symptoms should be those of allergy"


if __name__ == "__main__":
    test_incubation_days()
    test_human_compute_covid_properties()
    test_human_viral_load_for_day()
    test_human_cold_symptoms()
    test_human_flu_symptoms()
    test_human_allergies_symptoms()
