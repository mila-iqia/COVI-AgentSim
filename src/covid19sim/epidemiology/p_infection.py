import datetime
import numpy as np
from covid19sim.utils.constants import SECONDS_PER_DAY

def get_human_human_p_transmission(infector, infectors_infectiousness, infectee, social_contact_factor, contagion_knob, self, h):
    """
    Computes probability of virus transmission from infector to infectee.

    We use the model used here (section: Infection Dynamics)
        https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/covid19.md

    Specfically, probability of transmission is proportional to
        - susceptibility of the infectee: it is a proxy for the inverse of immunity to COVID
        - social factor of the location: this factor allows for a control in relative fraction of transmission that takes place at different locations
        - infectiousness of the infector - it serves as a proxy for probability of infecting someone based on viral load
    Further, probability of transmission is inversely proportional to
        - mean daily interactions of the infectee - (not explained well in the documents)

    Args:
        infector (covid19sim.human.Human):
        infectors_infectiousness (float):
        infectee (covid19sim.human.Human):
        social_contact_factor (float):
        contagion_knob (float):
        self (covid19sim.human.Human):
        h (covid19sim.human.Human):

    Returns:
        (float): probability of virus transmission
    """
    # probability of transmission
    rate_of_infection = infectee.normalized_susceptibility * social_contact_factor * 1 / infectee.mean_daily_interaction_age_group
    rate_of_infection *= infectors_infectiousness
    rate_of_infection *= contagion_knob
    p_infection = 1 - np.exp(-rate_of_infection)
    return p_infection

def get_environment_human_p_transmission(contamination_probability, human, environmental_infection_knob):
    """
    computes probability of virus transmission to human via environmental contamination.
    NOTE: the forumlation used here is completely experimental. We assume it to be proportional to
        - virus strength at the location
        - susceptibility of the person which acts as a proxy for inverse of immunity
    We further degrade this probability via reduction factors.

    Args:
        contamination_probability (float): current virus strength at a location. It is used as a proxy for probability.
        human (covid19sim.human.Human): `human` for whom we need to compute this probability of transmission.
        environmental_infection_knob (float): a global factor to calibrate probabilities such that total fraction of environmental transmission is as observed in the real world.

    Returns:
        (float): probability of environmetal contamination
    """
    p_infection = contamination_probability * human.normalized_susceptibility
    p_infection *= environmental_infection_knob
    return p_infection

def infectiousness_delta(human, t_near):
    """
    Computes area under the probability curve defined by infectiousness and time duration
    of self.env.timestamp and self.env.timestamp + delta_timestamp.
    Currently, it only takes the average of starting and ending probabilities.

    Args:
        t_near (float): time spent near another person in seconds

    Returns:
        area (float): area under the infectiousness curve is computed for this duration
    """

    if not human.is_infectious:
        return 0

    start_p = human.get_infectiousness_for_day(human.env.timestamp, human.is_infectious)
    end_p = human.get_infectiousness_for_day(human.env.timestamp + datetime.timedelta(seconds=t_near), human.is_infectious)
    area = (t_near / SECONDS_PER_DAY) * (start_p + end_p) / 2
    return area
