import typing
from typing import Iterable

import numpy as np

if typing.TYPE_CHECKING:
    from covid19sim.human import Human


def compute_prevalence(
    humans: Iterable["Human"],
    rng: np.random.RandomState,
    laplace_noise_scale: float = 0.1,
):
    all_risks = np.asarray(
        [human.risk for human in humans if not human.is_dead and human.has_app]
    )
    all_risks += rng.laplace(0, laplace_noise_scale, size=all_risks.shape)
    all_risks = np.abs(all_risks)
    prevalence = all_risks.mean()
    return prevalence, list(all_risks)


def get_recommendation_thresholds(
    prevalence: float,
    all_risks: Iterable[float],
    leeway: float,
    relative_fraction_orange_to_red: float,
    relative_fraction_yellow_to_red: float,
    eps: float = 1e-7,
):
    # First things first, we need to validate prevalence
    # (in order to not hit the assert below).
    # In other words, we must satisfy the constraint that:
    #   P * (1 + L) (1 + O2R + Y2R) < 1
    # where:
    #   P: Prevalence; L: Leeway;
    #   O2R: `relative_fraction_orange_to_red`; Y2R: `relative_fraction_yellow_to_red`
    max_prevalence = 1 / (
        (1 + leeway)
        * (1 + relative_fraction_orange_to_red + relative_fraction_yellow_to_red)
    )
    prevalence = np.clip(prevalence, a_max=max_prevalence - eps, a_min=0)
    # Compute the fraction of folks to put in red, orange, yellow, and green
    fraction_in_red = prevalence * (1 + leeway)
    fraction_in_orange = relative_fraction_orange_to_red * fraction_in_red
    fraction_in_yellow = relative_fraction_yellow_to_red * fraction_in_red
    fraction_in_green = 1 - (fraction_in_red + fraction_in_orange + fraction_in_yellow)

    assert fraction_in_green >= 0

    # Compute the percentiles of the risk level
    # (this is how we'll split the histogram)
    percentile_in_green = 100 * (fraction_in_green)
    percentile_in_yellow = percentile_in_green + (100 * fraction_in_yellow)
    percentile_in_orange = percentile_in_yellow + (100 * fraction_in_orange)

    # We have green_threshold < yellow_threshold < orange_threshold < 1.
    # We now split the histogram accordingly.
    green_threshold = np.percentile(all_risks, percentile_in_green)
    yellow_threshold = np.percentile(all_risks, percentile_in_yellow)
    orange_threshold = np.percentile(all_risks, percentile_in_orange)

    return green_threshold, yellow_threshold, orange_threshold
