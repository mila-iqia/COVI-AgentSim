"""
Holds basic functions to fit a curve to data.
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

np.random.seed(0)

class FittedFn(object):
    """
    Encapsulates all necessary functions to fit and compare different functions that fit x and y (assumed 1D).
    """
    def __init__(self):
        self.p0 = None # initial parameter value for this function
        self.fn = None # a callable function that takes parameters
        self.inv_fn = None # a callable function that takes parameters

        # initialized after the call to self.fit
        self.res = None
        self.parameters = None
        self.covariance = None
        self.stddev_parameters = None
        self.r_squared = None
        self._fit = False

    def fit(self, X, Y):
        """
        Optimize parameters of `self.fn` to find a relationship between X and Y using self.fn

        Args:
            X (np.array): x coordinate
            Y (np.array): corresponding y coordinate

        Returns:
            (FittedFn): a fitted function with self._fit = True
        """
        pass

    def find_x_for_y(self, y):
        """
        Finds x coordinate for which `self.fn` is y.

        Args:
            y (np.array): y coordinate

        Returns:
            (np.array): corresponding x coordinate
        """
        pass

    def evaluate_y_for_x(self, x):
        """
        Finds y coordinate for which `self.fn` is x.

        Args:
            x (np.array): x coordinate

        Returns:
            (np.array): corresponding y coordinate
        """
        pass

    def predict_y_using_sampled_fns(self, x):
        """
        Returns the predictions at `x` from fns that are sampled from `self.parameters` and corresponding `self.covriance`.

        Args:
            x (np.array): x coordinate at which stderr is to be evaluated

        Returns:
            (np.array): corresponding y coordinate
        """
        pass

    def stderr_for_x(self, x, analytical=True, return_var=False, n_samples=1):
        """
        Finds stderror (epistemic uncertainty) of predictions at `x` using random draws.

        Args:
            x (np.array): x coordinate at which stderr is to be evaluated
            return_var (bool): if True, return variance.
            analytical (bool): if True, computes analytical variance / stddev
            n_samples (int): number of samples to draw for estimation. Only used if analytical = False.

        Returns:
            (np.array): corresponding epistemic uncertainty at each x
        """
        pass

    def find_offset_and_stderr_at_y(self, y, other_fn, analytical=True):
        """
        Finds the difference in y coordinate of `self.fn` and `other_fn` at x where `self.fn = y`.

        Args:
            y (float):
            other_fn (FittedFn):
            analytical (bool): Default is True.

        Returns:
            advantage (float): mean difference in predictions of `other_fn` and `self` at x for which `self` is `y`
            stddev (float): standard deviation of advantage
            cdf (float): probablity that advantage is greater than 0
        """
        pass


class LinearFit(object):
    """
    Implements FittedFn of linear function.
    """
    def __init__(self, use_intercept=True):
        if use_intercept:
            self.p0 = [0, 0]
            self.fn = lambda x,m,c: m * x + c
            self.inv_fn = lambda y,m,c: (y - c) / m
        else:
            self.p0 = [0]
            self.fn = lambda x,m: m * x
            self.inv_fn = lambda y,m: y / m

        self._fit = False

    def fit(self, X, Y):
        assert len(X) == len(Y), "x and y are not of the same size."

        #
        self.parameters, self.covariance = curve_fit(f=self.fn, xdata=X, ydata=Y, p0=self.p0, bounds=(-np.inf, np.inf))
        self._fit = True
        self.stddev_parameters = np.sqrt(np.diag(self.covariance))

        #
        self.res = Y - self.evaluate_y_for_x(X)
        self.r_squared = 1 - np.sum(self.res ** 2) / np.sum((Y - Y.mean()) ** 2)

        return self

    def find_x_for_y(self, y):
        assert self._fit, "Function has not been fitted yet"
        return self.inv_fn(y, *self.parameters)

    def evaluate_y_for_x(self, x):
        assert self._fit, "Function has not been fitted yet"
        return self.fn(x, *self.parameters)

    def predict_y_using_sampled_fns(self, x, n_samples=1):
        pars = np.random.multivariate_normal(mean=self.parameters, cov=self.covariance, size=n_samples)
        ys = np.zeros((n_samples, x.shape[0]))
        for i, par in enumerate(pars):
            ys[i] = self.fn(x, *par)

        return ys

    def stderr_for_x(self, x, analytical=True, return_var=False, n_samples=1000):
        assert self._fit, "Function has not been fitted yet"
        if analytical:
            # E(y|x) = mx + c
            # Var(E(y|x)) = Var(m) * x**2 + Var(c) + 2 * x * Cov(m, c)
            var = self.covariance[0,0] * x**2 + self.covariance[1,1] +  2 * x * self.covariance[1,0]
            return var if return_var else np.sqrt(var)
        else:
            ys = self.predict_y_using_sampled_fns(x, n_samples)
            std = np.std(ys, axis=0)
            return std ** 2 if return_var else std

    def find_offset_and_stderr_at_y(self, y, other_fn, analytical=True):
        assert self._fit, "Function has not been fitted yet"
        # assert type(y) == float, f"expected float got {type(y)}"

        y = np.array([y])
        x = self.find_x_for_y(y) # find x such that mean self.fn is y
        assert abs(self.evaluate_y_for_x(x) - y) < 1e-4, f"Expected evaluation to be {y} but output is {self(x)}"
        if analytical:
            var_y1 = self.stderr_for_x(x, return_var=True, analytical=True)

            y2 = other_fn.evaluate_y_for_x(x)
            var_y2 = other_fn.stderr_for_x(x, return_var=True, analytical=True)

            offset = y2 - y
            stderr = np.sqrt(var_y2 + var_y1)
            cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)
        else:
            var_y1 = self.stderr_for_x(x, return_var=True, analytical=False)

            ys = other_fn.predict_y_using_sampled_fns(x, n_samples=1000)
            var_y2 = np.std(ys) ** 2

            offsets = ys - y
            offset = np.mean(offsets)
            stderr = np.sqrt(var_y2 + var_y1)
            cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)

        return offset.item(), stderr.item(), cdf.item()


class GPRFit(object):
    def __init__(self):
        pass
    
