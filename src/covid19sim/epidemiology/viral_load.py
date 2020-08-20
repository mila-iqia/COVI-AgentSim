import math
import numpy as np
from scipy.stats import gamma, truncnorm
from covid19sim.epidemiology.symptoms import _get_covid_progression, \
    MODERATE, SEVERE, EXTREMELY_SEVERE
from covid19sim.utils.constants import SECONDS_PER_DAY

def _sample_viral_load_gamma(rng, shape_mean=4.5, shape_std=.15, scale_mean=1., scale_std=.15):
    """
    This function samples the shape and scale of a gamma distribution, then returns it

    Args:
        rng ([type]): [description]
        shape_mean (float, optional): [description]. Defaults to 4.5.
        shape_std (float, optional): [description]. Defaults to .15.
        scale_mean ([type], optional): [description]. Defaults to 1..
        scale_std (float, optional): [description]. Defaults to .15.

    Returns:
        [type]: [description]
    """
    shape = rng.normal(shape_mean, shape_std)
    scale = rng.normal(scale_mean, scale_std)
    return gamma(shape, scale=scale)

def debug_infectiousness(infectiousness_onset_days, infectiousness_onset_days_wrt_incubation, incubation_days, scaling_factor):
    """Saves some numpy arrays to disk for later plotting and analysis"""
    try:
        incubation_days_np = np.load("incubation_days.npy")
        infectiousness_onset_days_wrt_incubation_np = np.load("infectiousness_onset_days_wrt_incubation.npy")
        infectiousness_onset_days_np = np.load("infectiousness_onset_days.npy")
    except Exception:
        incubation_days_np = []
        infectiousness_onset_days_wrt_incubation_np = []
        infectiousness_onset_days_np = []
    incubation_days_np = np.array(list(incubation_days_np) + [incubation_days])
    infectiousness_onset_days_wrt_incubation_np = np.array(list(infectiousness_onset_days_wrt_incubation_np) + [scaling_factor * infectiousness_onset_days_wrt_incubation])
    infectiousness_onset_days_np = np.array(list(infectiousness_onset_days_np) + [infectiousness_onset_days])
    np.save("incubation_days.npy", incubation_days_np)
    np.save("infectiousness_onset_days_wrt_incubation.npy", infectiousness_onset_days_wrt_incubation_np)
    np.save("infectiousness_onset_days.npy", infectiousness_onset_days_np)


def _get_disease_days(rng, conf, age, inflammatory_disease_level, debug_plot=False):
    """
    Defines viral load curve parameters.
    It is based on the study here https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf (Figure 1).

    We have used the same scale for the gamma distribution for all the parameters as fitted in the study here
        https://www.acpjournals.org/doi/10.7326/M20-0504 (Appendix Table 2)

    NOTE: Using gamma for all paramters is for the ease of computation.
    NOTE: Gamma distribution is only well supported in literature for incubation days

    Args:
        rng (np.random.RandomState): random number generator
        conf (dict): configuration dictionary
        age (float): age of human
        inflammatory_disease_level (int): based on count of inflammatory conditions.
    """
    # NOTE: references are in core.yaml alongside above parameters
    # All days count FROM EXPOSURE i.e. infection_timestamp

    PLATEAU_DURATION_CLIP_HIGH = conf.get("PLATEAU_DURATION_CLIP_HIGH")
    PLATEAU_DURATION_CLIP_LOW = conf.get("PLATEAU_DURATION_CLIP_LOW")
    PLATEAU_DURATION_MEAN = conf.get("PLATEAU_DURATION_MEAN")
    PLATEAU_DURATION_STD = conf.get("PLATEAU_DURATION_STD")

    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_AVG")
    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_STD")
    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_LOW = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_CLIP_LOW")
    INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_HIGH = conf.get("INFECTIOUSNESS_ONSET_DAYS_WRT_SYMPTOM_ONSET_CLIP_HIGH")

    INFECTIOUSNESS_PEAK_AVG = conf.get("INFECTIOUSNESS_PEAK_AVG")
    INFECTIOUSNESS_PEAK_STD = conf.get("INFECTIOUSNESS_PEAK_STD")
    INFECTIOUSNESS_PEAK_CLIP_HIGH = conf.get("INFECTIOUSNESS_PEAK_CLIP_HIGH")
    INFECTIOUSNESS_PEAK_CLIP_LOW = conf.get("INFECTIOUSNESS_PEAK_CLIP_LOW")

    RECOVERY_DAYS_AVG = conf.get("RECOVERY_DAYS_AVG")
    RECOVERY_STD = conf.get("RECOVERY_STD")
    RECOVERY_CLIP_LOW = conf.get("RECOVERY_CLIP_LOW")
    RECOVERY_CLIP_HIGH = conf.get("RECOVERY_CLIP_HIGH")

    # (no-source) assumption is that there is at least two days to remain exposed
    # Comparitively, we set infectiousness_onset_days to be at least one day to remain exposed
    # resample until we get into the correct range
    incubation_days = -math.inf
    while incubation_days < 2.0:

        # days after exposure when symptoms show up
        incubation_days = rng.gamma(
            shape=conf['INCUBATION_DAYS_GAMMA_SHAPE'],
            scale=conf['INCUBATION_DAYS_GAMMA_SCALE']
        )

    # days after exposure when viral shedding starts, i.e., person is infectious
    # resample until we get into the correct range in order to further truncate the range
    # (expert opinion) we assume that there is at least one day spent in the exposed state before becoming infectious
    infectiousness_onset_days_wrt_incubation = -math.inf

    # We scale the infectiousness onset days wrt the incubation days by a factor derived from the scale of the incubation days wrt to the shape parameter of the gamma
    scaling_factor = incubation_days / conf['INCUBATION_DAYS_GAMMA_SHAPE']
    while infectiousness_onset_days_wrt_incubation < 1.0:
        infectiousness_onset_days_wrt_incubation = scaling_factor * truncnorm((INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_LOW - INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG) / INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD,
                  (INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_CLIP_HIGH - INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG) / INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD,
                  loc=INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG,
                  scale=INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_STD).rvs(1, random_state=rng).item()
        infectiousness_onset_days = incubation_days - infectiousness_onset_days_wrt_incubation

    if debug_plot:
        debug_infectiousness(infectiousness_onset_days, infectiousness_onset_days_wrt_incubation, incubation_days, scaling_factor)

    # viral load peaks INFECTIOUSNESS_PEAK_AVG days before incubation days
    viral_load_peak_wrt_incubation_days = \
        truncnorm((INFECTIOUSNESS_PEAK_CLIP_LOW - INFECTIOUSNESS_PEAK_AVG) /
                  INFECTIOUSNESS_PEAK_STD,
                  (INFECTIOUSNESS_PEAK_CLIP_HIGH - INFECTIOUSNESS_PEAK_AVG) /
                  INFECTIOUSNESS_PEAK_STD,
                  loc=INFECTIOUSNESS_PEAK_AVG,
                  scale=INFECTIOUSNESS_PEAK_STD).rvs(1, random_state=rng).item()

    viral_load_peak = incubation_days - viral_load_peak_wrt_incubation_days

    # (no-source) assumption is that there is at least half a day after the infectiousness_onset_days
    viral_load_peak = max(infectiousness_onset_days + 0.5, viral_load_peak)

    viral_load_peak_wrt_incubation_days = incubation_days - viral_load_peak

    # (no-source) We assume that plateau start is equi-distant from the peak
    # infered from the curves in Figure 1 of the reference above
    plateau_start = incubation_days + viral_load_peak_wrt_incubation_days

    # (no-source) plateau duration is assumed to be of avg PLATEAU_DRATION_MEAN
    plateau_end = \
        plateau_start + \
        truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN) /
                  PLATEAU_DURATION_STD,
                  (PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) /
                  PLATEAU_DURATION_STD,
                  loc=PLATEAU_DURATION_MEAN,
                  scale=PLATEAU_DURATION_STD).rvs(1, random_state=rng).item()

    # recovery is often quoted with respect to the incubation days
    # so we add it here with respect to the plateau end.
    RECOVERY_WRT_PLATEAU_END_AVG = RECOVERY_DAYS_AVG - PLATEAU_DURATION_MEAN - INFECTIOUSNESS_ONSET_WRT_SYMPTOM_ONSET_AVG
    recovery_days = \
        plateau_end + \
        truncnorm((RECOVERY_CLIP_LOW - RECOVERY_WRT_PLATEAU_END_AVG) /
                  RECOVERY_STD,
                  (RECOVERY_CLIP_HIGH - RECOVERY_WRT_PLATEAU_END_AVG) /
                  RECOVERY_STD,
                  loc=RECOVERY_WRT_PLATEAU_END_AVG,
                  scale=RECOVERY_STD).rvs(1, random_state=rng).item()

    # Time to recover is proportional to age
    # based on hospitalization data (biased towards older people) https://pubs.rsna.org/doi/10.1148/radiol.2020200370
    # (no-source) it adds dependency of recovery days on age
    recovery_days += age/40

    # viral load height. There are two parameters here -
    # peak - peak of the viral load curve
    # plateau - plateau of the viral load curve
    # max: 130/200 + 3/3.5 = 2.5, scales the base to [0-1]
    # Older people and those with inflammatory diseases have higher viral load
    # https://www.medrxiv.org/content/10.1101/2020.04.10.20061325v2.full.pdf
    # TODO : make it dependent on initial viral load
    # (no-source) dependence on age vs inflammatory_disease_count
    # base = conf['AGE_FACTOR_VIRAL_LOAD_HEIGHT'] * age/200 + conf['INFLAMMATORY_DISEASE_FACTOR_VIRAL_LOAD_HEIGHT'] * np.exp(-inflammatory_disease_level/3)
    base = 1.0

    # as long as min and max viral load are [0-1], this will be [0-1]
    peak_height = rng.uniform(conf['MIN_VIRAL_LOAD_PEAK_HEIGHT'], conf['MAX_VIRAL_LOAD_PEAK_HEIGHT']) * base

    # as long as min and max viral load are [0-1], this will be [0-1]
    plateau_height = peak_height * rng.uniform(conf['MIN_MULTIPLIER_PLATEAU_HEIGHT'], conf['MAX_MULTIPLIER_PLATEAU_HEIGHT'])

    assert peak_height != 0, f"viral load of peak of 0 sampled age:{age}"
    return infectiousness_onset_days, viral_load_peak, incubation_days, plateau_start, plateau_end, recovery_days, peak_height, plateau_height


def _sample_viral_load_piecewise(rng, plateau_start, initial_viral_load=0, age=40, conf={}):
    """
    This function samples a piece-wise linear viral load model which increases, plateaus, and drops.

    Args:
        rng (np.random.RandomState): random number generator
        plateau_start: start of the plateau with respect to infectiousness_onset_days
        initial_viral_load (int, optional): unused
        age (int, optional): age of the person. Defaults to 40.

    Returns:
        plateau_height (float): height of the plateau, i.e., viral load at its peak
        plateau_end (float): days after beign infectious when the plateau ends
        recovered (float): days after being infectious when the viral load is assumed to be ineffective (not necessarily 0)
    """

    MAX_VIRAL_LOAD = conf.get("MAX_VIRAL_LOAD")
    MIN_VIRAL_LOAD = conf.get("MIN_VIRAL_LOAD")
    PLATEAU_DURATION_CLIP_HIGH = conf.get("PLATEAU_DURATION_CLIP_HIGH")
    PLATEAU_DURATION_CLIP_LOW = conf.get("PLATEAU_DURATION_CLIP_LOW")
    PLATEAU_DURATION_MEAN = conf.get("PLATEAU_DURATION_MEAN")
    PLATEAU_DURATION_STD = conf.get("PLATEAU_DURATION_STD")
    RECOVERY_CLIP_HIGH = conf.get("RECOVERY_CLIP_HIGH")
    RECOVERY_CLIP_LOW = conf.get("RECOVERY_CLIP_LOW")
    RECOVERY_MEAN = conf.get("RECOVERY_MEAN")
    RECOVERY_STD = conf.get("RECOVERY_STD")
    VIRAL_LOAD_RECOVERY_FACTOR = conf.get("VIRAL_LOAD_RECOVERY_FACTOR")

    plateau_end = plateau_start + truncnorm((PLATEAU_DURATION_CLIP_LOW - PLATEAU_DURATION_MEAN)/PLATEAU_DURATION_STD,
                                            (PLATEAU_DURATION_CLIP_HIGH - PLATEAU_DURATION_MEAN) / PLATEAU_DURATION_STD,
                                            loc=PLATEAU_DURATION_MEAN, scale=PLATEAU_DURATION_STD).rvs(1, random_state=rng)

    recovered = plateau_end + ((age/10)-1) # age is a determining factor for the recovery time
    recovered = recovered + initial_viral_load * VIRAL_LOAD_RECOVERY_FACTOR \
                          + truncnorm((RECOVERY_CLIP_LOW - RECOVERY_MEAN) / RECOVERY_STD,
                                        (RECOVERY_CLIP_HIGH - RECOVERY_MEAN) / RECOVERY_STD,
                                        loc=RECOVERY_MEAN, scale=RECOVERY_STD).rvs(1, random_state=rng)

    base = age/200 # peak viral load varies linearly with age
    plateau_height = rng.uniform(base + MIN_VIRAL_LOAD, base + MAX_VIRAL_LOAD)
    return plateau_height, plateau_end.item(), recovered.item()


def compute_covid_properties(human):
    """
    Pre-computes viral load curve.
    Specifically, characteristics of viral load plateau curve, i.e., height, start/end of plateau,
    start of infectiousness and when the symptom will show up.
    """
    # NOTE: all the days returned here are with respect to exposure day
    human.infectiousness_onset_days, human.viral_load_peak_start, \
    human.incubation_days, human.viral_load_plateau_start, \
    human.viral_load_plateau_end, human.recovery_days, \
    human.viral_load_peak_height, human.viral_load_plateau_height = _get_disease_days(human.rng, human.conf, human.age,
                                                                                    human.inflammatory_disease_level)

    # for ease of calculation, make viral load parameters relative to infectiousness onset
    human.viral_load_peak_start -= human.infectiousness_onset_days
    human.viral_load_plateau_start -= human.infectiousness_onset_days
    human.viral_load_plateau_end -= human.infectiousness_onset_days

    # precompute peak-plateau slope
    denominator = (human.viral_load_plateau_start - human.viral_load_peak_start)
    numerator = human.viral_load_peak_height - human.viral_load_plateau_height
    human.peak_plateau_slope = numerator / denominator
    assert human.peak_plateau_slope >= 0, f"viral load should decrease after peak. peak:{human.viral_load_peak_height} plateau height:{human.viral_load_plateau_height}"

    # percomupte plateau-end - recovery slope (should be negative because it is decreasing)
    numerator = human.viral_load_plateau_height
    denominator = human.recovery_days - (human.viral_load_plateau_end + human.infectiousness_onset_days)
    human.plateau_end_recovery_slope = numerator / denominator
    assert human.plateau_end_recovery_slope >= 0, f"slopes are assumed to be positive for ease of calculation"

    human.covid_progression = []
    if not human.is_asymptomatic:
        human.covid_progression = _get_covid_progression(human.initial_viral_load, human.viral_load_plateau_start,
                                                        human.viral_load_plateau_end,
                                                        human.recovery_days, age=human.age,
                                                        incubation_days=human.incubation_days,
                                                        infectiousness_onset_days=human.infectiousness_onset_days,
                                                        really_sick=human.can_get_really_sick,
                                                        extremely_sick=human.can_get_extremely_sick,
                                                        rng=human.rng,
                                                        preexisting_conditions=human.preexisting_conditions,
                                                        carefulness=human.carefulness)

    all_symptoms = set(symptom for symptoms_per_day in human.covid_progression for symptom in symptoms_per_day)
    # infection ratios
    if human.is_asymptomatic:
        human.infection_ratio = human.conf['ASYMPTOMATIC_INFECTION_RATIO']

    elif sum(x in all_symptoms for x in [MODERATE, SEVERE, EXTREMELY_SEVERE]) > 0:
        human.infection_ratio = 1.0

    else:
        human.infection_ratio = human.conf['MILD_INFECTION_RATIO']

    if hasattr(human.city, "tracker"):  # some tests are running with dummy cities that don't track anything
        human.city.tracker.track_covid_properties(human)


def viral_load_for_day(human, timestamp):
    """ Calculates the elapsed time since infection, returning this person's current viral load"""

    if not human.has_covid:
        return 0.

    # calculates the time since infection in days
    if isinstance(timestamp, (int, float)):
        days_infectious = (timestamp - human.ts_covid19_infection) / SECONDS_PER_DAY - \
                          human.infectiousness_onset_days
    else:
        days_infectious = (timestamp - human.infection_timestamp).total_seconds() / SECONDS_PER_DAY - \
                          human.infectiousness_onset_days

    if days_infectious < 0:
        return 0.

    # Rising to peak
    if days_infectious < human.viral_load_peak_start:
        cur_viral_load = human.viral_load_peak_height * days_infectious / (human.viral_load_peak_start)

    # Descending to plateau from peak
    elif days_infectious < human.viral_load_plateau_start:
        days_since_peak = days_infectious - human.viral_load_peak_start
        cur_viral_load = human.viral_load_peak_height - human.peak_plateau_slope * days_since_peak

    # plateau duration
    elif days_infectious < human.viral_load_plateau_end:
        cur_viral_load = human.viral_load_plateau_height

    # during recovery
    else:
        days_since_plateau_end = days_infectious - human.viral_load_plateau_end
        cur_viral_load = human.viral_load_plateau_height - human.plateau_end_recovery_slope * days_since_plateau_end
        cur_viral_load = max(0, cur_viral_load) # clip it at 0

    assert 0 <= cur_viral_load <= 1, f"effective viral load out of bounds. viral load:{cur_viral_load} plateau_end:{days_since_plateau_end}"

    return cur_viral_load
