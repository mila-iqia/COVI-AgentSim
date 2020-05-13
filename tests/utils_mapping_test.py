import pytest
import numpy as np

from covid19sim.utils import probas_to_risk_mapping, proba_to_risk_fn


probabilities = np.array([
    [[0.89616345, 0.87178723, 0.41788727, 0.98024759, 0.89319564, 0.86549061, 0.88135983],
     [0.68489395, 0.43069957, 0.52555108, 0.83728212, 0.21557683, 0.53409188, 0.36770284],
     [0.50637388, 0.62591151, 0.92860176, 0.20259845, 0.25901962, 0.45586679, 0.42916668],
     [0.42175074, 0.78235259, 0.71177176, 0.34499354, 0.86636619, 0.72717452, 0.89602893],
     [0.46214522, 0.34554782, 0.61159501, 0.00652984, 0.8631294 , 0.06814696, 0.65331657]],
    [[0.34261144, 0.83686459, 0.75594126, 0.7139314 , 0.23626577, 0.01528256, 0.57191369],
     [0.88894116, 0.53144062, 0.51125875, 0.44913223, 0.13028579, 0.27107773, 0.10572309],
     [0.27818336, 0.43706775, 0.23721451, 0.24469225, 0.6985641 , 0.91273931, 0.87708491],
     [0.67135535, 0.09192728, 0.50558772, 0.84059843, 0.2614552 , 0.86062312, 0.59974303],
     [0.67021344, 0.2276843 , 0.6095271 , 0.41824453, 0.12363199, 0.97394058, 0.28072422]],
    [[0.12616978, 0.73155964, 0.11994702, 0.98259938, 0.01387591, 0.50996332, 0.33490804],
     [0.14223107, 0.9957335 , 0.392577  , 0.71522121, 0.58123311, 0.94698178, 0.48345278],
     [0.11031009, 0.65444631, 0.91594683, 0.12746956, 0.29189684, 0.27521415, 0.10702706],
     [0.56123148, 0.45419889, 0.63998031, 0.06696658, 0.82367792, 0.38114687, 0.02700863],
     [0.53971996, 0.15226332, 0.60115902, 0.56032635, 0.90137517, 0.40213379, 0.32293645]]])


@pytest.mark.parametrize('num_bins', [4, 16])
@pytest.mark.parametrize('lower_cutoff', [None, 0.01])
@pytest.mark.parametrize('upper_cutoff', [None, 0.99])
def test_probas_to_risk_mapping(num_bins, lower_cutoff, upper_cutoff):
    mapping = probas_to_risk_mapping(probabilities,
                                     num_bins=num_bins,
                                     lower_cutoff=lower_cutoff,
                                     upper_cutoff=upper_cutoff)

    # The mapping has length num_bins + 1
    assert mapping.shape == (num_bins + 1,)

    # First value is always 0, last value is always 1
    assert mapping[0] == 0.
    assert mapping[-1] == 1.

    # The mapping is monotonically increasing
    assert np.all(np.diff(mapping) >= 0)


@pytest.mark.parametrize('num_bins', [4, 16])
def test_probas_to_risk_mapping_approx_uniform_bins(num_bins):
    mapping = probas_to_risk_mapping(probabilities, num_bins=num_bins)
    histogram, _ = np.histogram(probabilities, bins=mapping)

    assert histogram.shape == (num_bins,)
    # The number of values in each bin are at most 1 apart from one another
    assert np.ptp(histogram) <= 1


@pytest.mark.parametrize('num_bins', [4, 16])
def test_proba_to_risk_fn(num_bins):
    # Dummy mapping (uniform)
    mapping = np.linspace(0, 1, num_bins + 1)
    proba_to_risk = proba_to_risk_fn(mapping)

    risk_levels = proba_to_risk(probabilities)

    # The mapping function preserves the shape
    assert risk_levels.shape == probabilities.shape
    assert risk_levels.dtype == np.int_

    # The risk levels should be in [0, num_bins - 1]
    assert np.all(risk_levels >= 0)
    assert np.all(risk_levels < num_bins)
