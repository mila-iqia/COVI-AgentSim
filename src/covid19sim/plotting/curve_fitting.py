"""
Holds basic functions to fit a curve to data.
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

np.random.seed(0)

def _linear(x, m, c):
    return m * x + c

def _linear_with_no_intercept(x, m):
    return lambda x: _linear(x, m, c=0)

def _linear_inverse(y, m, c):
    return (y - c) / m

def _linear_with_no_intercept_inverse(y, m):
    return y / m

def get_inverse_function(fn, *pars):
    """
    Maps fn to the inverse of fn.

    Args:
        fn (function): For example, `_linear`

    Returns:
        fn (function): Corresponding inverse. For example, `_linear_inverse`
    """
    if fn.__name__ == "_linear":
        inv_fn = _linear_inverse

    elif fn.__name__ == "_linear_with_no_intercept":
        inv_fn = _linear_with_no_intercept_inverse

    else:
        raise ValueError(f"Unknown function to find inverse for: {fn.__name__}")

    return inv_fn

def get_fitted_fn(x, y, fn):
    """
    Fits a function between x and y of kind `kind`.

    Args:
        x (np.array): 1D array of values to use as input to the predictor
        y (np.array): 1D array of y-values to predict from x
        fn (function): a function that accepts an np.array and corresponding parameters. Examples - `_linear`

    Returns:
        fn_handle (lambda): A function that takes in x and outputs y
        residuals (np.array): residuals from the fit
        r_squared (float): a value representing fitness. Its None when kind is not linear
        parameters (tuple): tuple of parameters and stddev of parameters
    """
    assert len(x) == len(y), "x and y are not of the same size"

    #
    pars, cov = curve_fit(f=_linear, xdata=x, ydata=y, p0=[0, 0], bounds=(-np.inf, np.inf))
    stdevs = np.sqrt(np.diag(cov))

    #
    res = y - fn(x, *pars)
    r_squared = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)

    #
    fn_handle_inverse = get_inverse_function(fn)
    return fn, res, r_squared, (pars, stdevs, cov), fn_handle_inverse

def get_offset_and_stddev_from_random_draws(reference_fn, reference_inv_fn, reference_stats, other_method_fn, other_method_inv_fn, other_method_stats):
    """
    Computes advantage of reference over other_method with confidence bounds.

    Args:
        reference_fn (function):
        reference_inv_fn (function):
        reference_stats (dict): {'res':residuals (np.array), 'parameters':, 'stddev_parameters': (), 'covariance': ()}
        other_method_fn (function):
        other_method_inv_fn (function):
        other_method_stats (dict):

    Returns:
        advantage (float):
        stddev (float):
        cdf (float):
    """
    # from random draws
    N = 1000
    pars1 = reference_stats['parameters']
    cov1 = reference_stats['covariance']
    pars = np.random.multivariate_normal(mean=pars1, cov=cov1, size=N)
    xs = [reference_inv_fn(1.0, *par) for par in pars]

    #
    pars2 = other_method_stats['parameters']
    cov2 = other_method_stats['covariance']
    pars = np.random.multivariate_normal(mean=pars2, cov=cov2, size=N)
    ys = [other_method_fn(x, *par) for x in xs for par in pars]

    offset = np.mean(ys) - 1.0
    stderr = stats.sem(ys)
    cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)
    return offset, stderr, cdf


def get_bounds_of_fitted_fn(function, x, parameters, covariance):
    """

    Args:
        function (function): function to be used for prediction using x
        x (np.array): value of x for which prediction is to be made
        parameters (np.array): mean value of parameters
        covariance (np.array): 2D matrix representing covariance in parameters

    Returns:
        upper_bound (np.array): Max prediction in all evalautions for each x
        lower_bound (np.array): Min prediction in all evalautions for each x
    """
    N = 1000
    pars = np.random.multivariate_normal(mean=parameters, cov=covariance, size=N)
    ys = np.zeros((N, x.shape[0]))
    for i, par in enumerate(pars):
        ys[i] = function(x, *par)

    return ys.max(axis=0), ys.min(axis=0)
