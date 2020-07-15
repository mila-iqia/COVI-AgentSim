import typing
from typing import Iterable

import numpy as np

if typing.TYPE_CHECKING:
    from covid19sim.human import Human


def compute_prevalence(
    humans: Iterable["Human"], risk_level_threshold: float,
):
    infectious_count = 0
    total_count = 0
    all_risk_levels = []
    for human in humans:
        if human.is_dead or not human.has_app:
            continue
        current_risk_level = human.risk_level
        all_risk_levels.append(current_risk_level)
        # fmt: off
        total_count += 1
        infectious_count += (1 if current_risk_level > risk_level_threshold else 0)
        # fmt: on
    if total_count == 0:
        prevalence = None
    else:
        prevalence = infectious_count / total_count
    return dict(prevalence=prevalence, all_risk_levels=all_risk_levels)


def get_recommendation_thresholds(
    prevalence: float,
    all_risk_levels: Iterable[int],
    leeway: float,
    relative_fraction_orange_to_red: float,
    relative_fraction_yellow_to_red: float,
):
    # Compute the fraction of folks to put in red, orange, yellow, and green
    fraction_in_red = prevalence * (1 + leeway)
    fraction_in_orange = relative_fraction_orange_to_red * fraction_in_red
    fraction_in_yellow = relative_fraction_yellow_to_red * fraction_in_red
    fraction_in_green = 1 - (fraction_in_red + fraction_in_orange + fraction_in_yellow)

    assert fraction_in_green >= 0

    # Compute the percentiles of the risk level
    # (this is how we'll split the histogram)
    percentile_in_green = 100 * (fraction_in_green)
    percentile_in_yellow = 100 * (percentile_in_green + fraction_in_yellow)
    percentile_in_orange = 100 * (percentile_in_yellow + fraction_in_orange)

    # We have green_threshold < yellow_threshold < orange_threshold < 1.
    # We now split the histogram accordingly.
    green_threshold = np.percentile(all_risk_levels, percentile_in_green)
    yellow_threshold = np.percentile(all_risk_levels, percentile_in_yellow)
    orange_threshold = np.percentile(all_risk_levels, percentile_in_orange)

    return green_threshold, yellow_threshold, orange_threshold
