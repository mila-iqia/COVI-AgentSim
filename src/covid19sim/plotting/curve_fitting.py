"""
Holds basic functions to fit a curve to data.
"""
import numpy as np
from scipy.optimize import curve_fit

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

    return lambda y: inv_fn(y, *pars)

def get_fitted_fn(x, y, fn):
    """
    Fits a function between x and y of kind `kind`.

    Args:
        x (np.array): 1D array of values to use as input to the predictor
        y (np.array): 1D array of y-values to predict from x
        fn (function): a function that accepts an np.array. Examples - `_linear`

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
    fn_handle = lambda x: fn(x, *pars)
    fn_handle_inverse = get_inverse_function(fn, *pars)
    return fn_handle, res, r_squared, (pars, stdevs), fn_handle_inverse
