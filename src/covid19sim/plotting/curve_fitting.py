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

def get_stderr_of_fitted_fn_from_random_draws(function, x, parameters, covariance):
    """
    Computes stddev of function i.e. Var(E(y|x)) evaluated at x using random draws.

    Args:
        function (function): function to be used for prediction using x
        x (np.array): value of x for which prediction is to be made
        parameters (np.array): mean value of parameters
        covariance (np.array): 2D matrix representing covariance in parameters

    Returns:
        stderr (np.array): stderr at each x
    """
    N = 1000
    pars = np.random.multivariate_normal(mean=parameters, cov=covariance, size=N)
    ys = np.zeros((N, x.shape[0]))
    for i, par in enumerate(pars):
        ys[i] = function(x, *par)

    return np.std(ys, axis=0)

def get_stderr_of_fitted_fn_analytical(function, x, parameters, covariance, return_var=False):
    """
    Computes stddev of function i.e. Var(E(y|x)) evaluated at x anlytically.

    Args:
        function (function): function to be used for prediction using x
        x (np.array): value of x for which prediction is to be made
        parameters (np.array): mean value of parameters
        covariance (np.array): 2D matrix representing covariance in parameters
        return_var (bool): returns variance if True.

    Returns:
        stderr (np.array): stderr at each x
    """
    assert function.__name__ == "_linear", f"Do not know how to compuate variance anlytically for {function.__name__}"
    # E(y|x) = mx + c
    # Var(E(y|x)) = Var(m) * x**2 + Var(c) + 2 * x * Cov(m, c)
    var = covariance[0,0] * x**2 + covariance[1,1] +  2 * x * covariance[1,0]
    return var if return_var else np.sqrt(var)

def get_offset_and_stddev_from_random_draws(reference_fn, reference_inv_fn, reference_stats, other_method_fn, other_method_inv_fn, other_method_stats):
    """
    Computes Var(E(y|x)) by random sampling to estimate advantages of reference over other_method with confidence bounds.

    Args:
        reference_fn (function): function to which other function is to be compared. x for R = 1 is dervied from this function
        reference_inv_fn (function): an inverse of `reference_fn`
        reference_stats (dict): {'res':residuals (np.array), 'parameters': (np.array), 'stddev_parameters': (np.array), 'covariance': (no.array) 2D matrix}
        other_method_fn (function): function which is to be compared to reference fn
        other_method_inv_fn (function): an inverse of `other_method_fn`
        other_method_stats (dict): {'res':residuals (np.array), 'parameters': (np.array), 'stddev_parameters': (np.array), 'covariance': (no.array) 2D matrix}

    Returns:
        advantage (float): mean difference in predictions of `other_method_fn` and `reference_fn` at x for which `reference_fn` is 1.0
        stddev (float): standard deviation of advantage
        cdf (float): probablity that advantage is greater than 0
    """
    # from random draws
    N = 1000
    pars1 = reference_stats['parameters']
    x = reference_inv_fn(1.0, *pars1)

    #
    pars2 = other_method_stats['parameters']
    cov2 = other_method_stats['covariance']
    pars = np.random.multivariate_normal(mean=pars2, cov=cov2, size=N)
    ys = np.array([other_method_fn(x, *par) for par in pars])

    offsets = ys - 1.0
    offset = np.mean(offsets)
    stderr = np.std(offsets)
    cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)
    return offset, stderr, cdf

def get_offset_and_stddev_analytical(reference_fn, reference_inv_fn, reference_stats, other_method_fn, other_method_inv_fn, other_method_stats):
    """
    Computes Var(E(y|x)) analytically to estimate advantages of reference over other_method with confidence bounds.
    Following predictors are allowed - linear, .

    Args:
        reference_fn (function): function to which other function is to be compared. x for R = 1 is dervied from this function
        reference_inv_fn (function): an inverse of `reference_fn`
        reference_stats (dict): {'res':residuals (np.array), 'parameters': (np.array), 'stddev_parameters': (np.array), 'covariance': (no.array) 2D matrix}
        other_method_fn (function): function which is to be compared to reference fn
        other_method_inv_fn (function): an inverse of `other_method_fn`
        other_method_stats (dict): {'res':residuals (np.array), 'parameters': (np.array), 'stddev_parameters': (np.array), 'covariance': (no.array) 2D matrix}

    Returns:
        advantage (float): mean difference in predictions of `other_method_fn` and `reference_fn` at x for which `reference_fn` is 1.0
        stddev (float): standard deviation of advantage
        cdf (float): probablity that advantage is greater than 0
    """
    assert reference_fn.__name__ == "_linear", f"Don't know how to compute variance analytically for {reference_fn.__name__} function"
    assert other_method_fn.__name__ == "_linear", f"Don't know how to compute variance analytically for {other_method_fn.__name__} function"

    pars1 = reference_stats['parameters']
    x = reference_inv_fn(1.0, *pars1)
    y1 = reference_fn(x, *reference_stats['parameters'])
    assert y1 == 1.0, "incorrect inverse function"

    cov1 = reference_stats['covariance']
    var_y1 = get_stderr_of_fitted_fn_analytical(reference_fn, x, pars1, cov1, return_var=True)

    pars2 = other_method_stats['parameters']
    y2 = other_method_fn(x, *pars2)
    cov2 = other_method_stats['covariance']
    var_y2 = get_stderr_of_fitted_fn_analytical(other_method_fn, x, pars2, cov2, return_var=True)

    offset = y2 - y1
    stderr = np.sqrt(var_y2 + var_y1)
    cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)
    return offset, stderr, cdf
