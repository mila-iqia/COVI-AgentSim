import datetime
import numpy as np


def get_p_infection(infector, infectors_infectiousness, infectee, social_contact_factor, contagion_knob, mask_efficacy_factor, hygiene_efficacy_factor, self, h):
    # probability of transmission
    # It is similar to Oxford COVID-19 model described in Section 4.
    rate_of_infection = infectee.normalized_susceptibility * social_contact_factor * 1 / infectee.mean_daily_interaction_age_group
    rate_of_infection *= infectors_infectiousness
    rate_of_infection *= contagion_knob
    p_infection = 1 - np.exp(-rate_of_infection)

    # factors that can reduce probability of transmission.
    # (no-source) How to reduce the transmission probability mathematically?
    mask_efficacy = (self.mask_efficacy + h.mask_efficacy)
    # mask_efficacy = p_infection - infector.mask_efficacy * p_infection - infectee.mask_efficacy * p_infection
    hygiene_efficacy = self.hygiene + h.hygiene
    reduction_factor = mask_efficacy * mask_efficacy_factor + hygiene_efficacy * hygiene_efficacy_factor
    p_infection *= np.exp(-reduction_factor)
    return p_infection


def infectiousness_delta(human, t_near):
    """
    Computes area under the probability curve defined by infectiousness and time duration
    of self.env.timestamp and self.env.timestamp + delta_timestamp.
    Currently, it only takes the average of starting and ending probabilities.

    Args:
        t_near (float): time spent near another person in minutes

    Returns:
        area (float): area under the infectiousness curve is computed for this duration
    """

    if not human.is_infectious:
        return 0

    start_p = human.get_infectiousness_for_day(human.env.timestamp, human.is_infectious)
    end_p = human.get_infectiousness_for_day(human.env.timestamp + datetime.timedelta(minutes=t_near), human.is_infectious)
    area = t_near / (24 * 60) * (start_p + end_p) / 2
    return area
