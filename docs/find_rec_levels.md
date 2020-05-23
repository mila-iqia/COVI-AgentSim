# Documentation for `covid19sim/other/find_rec_level.py`

## Finding the best recommendation levels

This script finds the optimal recommendation level mapping used in `covid19sim`.

The idea is: given some ground-truth, unobserved data which we can get from the simulator, what is the best way to map `risk` or `risk_level` to a `rec_level`, using thresholds.

Given that `risk` is continuous but we're only looking for 3 thresholds (for 4 recommendation levels) *and* they should be ordered, the search space isn't too large and a naïve random sampling strategy should be enough to find the best mapping.

On the other hand, `risk_level` is both discrete and small (in `[0 .. 15]`), we can therefore do a full grid-search (from `[0, 1, 2]` to `[13, 14, 15]`).

Doing one or the other depends on `--risk_level`.

## Best?

We need to define what we're optimizing for. The script currently supports 4 criteria: F1-scores (micro, macro, weighted [[?](https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin)]) and custom error rate. This is defined by the `--score` flag.

The error rate is saying: for each sample that I misclassified, what is the cost of this misclassification (`costs_for_category`), as a function of the true, underlying category (`get_category(...)`). If all values in `costs_for_category` are equal, the script will care equally for all misclassifications, but we might care more for some other.

`/!\` The `score` we choose is not an *objective function* in the sense that it's not optimized for in any other way than random or grid search over its 3 parameters (*i.e.* thresholds)