"""
Holds basic functions to fit a curve to data.
"""
import gpflow
import warnings
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from gpflow.ci_utils import ci_niter
from tensorflow_probability import distributions as tfd

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
f64 = gpflow.utilities.to_default_float # convert to float64 for tfp to play nicely with gpflow in 64

tf.random.set_seed(123)
np.random.seed(0)
warnings.filterwarnings("ignore")

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

    def find_offset_and_stderr_at_x(self, x, other_fn, analytical=True, check_y=None):
        """
        Finds the difference in y coordinate of `self.fn` and `other_fn` at x where `self.fn = y`.

        Args:
            x (float): x co-ordinate at which this offset (advantage) wrt other_fn is to be computed
            other_fn (FittedFn): other function which is being compared to
            analytical (bool): If variance to be computed is analytical. Default is True.
            check_y (float): If not None, assertion is used to check correctness of the point lying on FittedFn.

        Returns:
            advantage (float): mean difference in predictions of `other_fn` and `self` at x for which `self` is `y`
            stddev (float): standard deviation of advantage
            cdf (float): probablity that advantage is greater than 0
        """
        pass

    def save(self, path):
        """
        Saves the necessary function to `path`

        Args:
            path (str): path where the model needs to be saved
        """
        pass

    def load(self, path):
        """
        Loads the saved model from `path`

        Args:
            path (str): path where model is saved
        """
        pass


class LinearFit(FittedFn):
    """
    Implements FittedFn of linear function.
    """
    def __init__(self, use_intercept=True):
        super().__init__()
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

    def find_offset_and_stderr_at_x(self, x, other_fn, analytical=True, check_y=None):
        assert self._fit, "Function has not been fitted yet"

        x = np.array([x])
        y = self.evaluate_y_for_x(x)
        if check_y is not None:
            assert abs(y - check_y) < 1e-4, f"Evaluated y are not equal. Evaluated: {y}. Expected: {check_y}"

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

            offset = np.mean(ys) - y
            stderr = np.sqrt(var_y2 + var_y1)
            cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)

        return offset.item(), stderr.item(), cdf.item()


class GPRFit(FittedFn):
    """
    Implements FittedFn using GP regression.
    Used from the tutorial here - https://gpflow.readthedocs.io/en/develop/notebooks/advanced/mcmc.html
    """
    def __init__(self, assume_linear_mean=False):
        super().__init__()
        self.kernel = gpflow.kernels.Matern32(variance=1, lengthscales=1)
        self.mean_function = None
        if assume_linear_mean:
            self.mean_function = gpflow.mean_functions.Linear()

        # number of samples of E(y|x) model to sample from the final posterior in `self.fit`
        self.num_samples = 1000
        self.using_saved_model = False

    def fit(self, X, Y):
        assert not self.using_saved_model, "Saved model's predict_f has been overwritten. Don't know the consequences of fitting to new data, so this function is not allowed at the moment."
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        optimizer = gpflow.optimizers.Scipy()
        self.model = gpflow.models.GPR(data=(X, Y), kernel=self.kernel, mean_function=self.mean_function)

        # initialize the model to the maximum likelihood solution.
        optimizer.minimize(
            self.model.training_loss,
            variables=self.model.trainable_variables,
            options=dict(disp=False, maxiter=100)
        )

        self._fit = True

        # fitness
        self.res = Y - self.evaluate_y_for_x(X)
        self.r_squared = 1 - np.sum(self.res ** 2) / np.sum((Y - Y.mean()) ** 2)

        return self

    def find_x_for_y(self, y):
        assert self._fit or self.using_saved_model, "Function has not been fitted yet"
        y = self.reformat_input(y)
        xx = np.linspace(2, 10, 100000).reshape(-1, 1)
        yy = self.evaluate_y_for_x(xx)
        x1 = xx[np.argmin(np.abs(yy - y))]
        return x1

    def evaluate_y_for_x(self, x):
        assert self._fit or self.using_saved_model, "Function has not been fitted yet"

        x = self.reformat_input(x)
        return self.model.predict_f(x)[0].numpy().reshape(-1)

    def predict_y_using_sampled_fns(self, x, n_samples=1):
        assert self._fit or self.using_saved_model, "Function has not been fitted yet"
        assert self.samples is not None, "No posterior samples found. "

        x = self.reformat_input(x)
        ys = np.zeros((n_samples, x.shape[0]))
        original_state = self.hmc_helper.current_state
        for i in range(0, n_samples):
            for var, var_samples in zip(self.hmc_helper.current_state, self.samples):
                var.assign(var_samples[i])
            f = self.model.predict_f(x)[0].numpy()
            ys[i, :] = f.reshape(-1)

        # revert back the values to the original function
        for var, prev_value in zip(self.hmc_helper.current_state, original_state):
            var.assign(prev_value)

        return ys

    def stderr_for_x(self, x, analytical=True, return_var=False, n_samples=1000):
        assert self._fit or self.using_saved_model, "Function has not been fitted yet"
        x = self.reformat_input(x)
        if analytical:
            mean, var = self.model.predict_f(x)
            var = var.numpy().reshape(-1)
            return var if return_var else np.sqrt(var)
        else:
            warnings.warn("Empirical stderr for GP Regression is expensive!")
            ys = self.predict_y_using_sampled_fns(x, self.num_samples)
            std = np.std(ys, axis=0)
            return std ** 2 if return_var else std

    def find_offset_and_stderr_at_x(self, x, other_fn, analytical=False, check_y=None):
        assert self._fit or self.using_saved_model, "Function has not been fitted yet"

        x = self.reformat_input(x)
        y = self.evaluate_y_for_x(x)
        if check_y is not None:
            assert abs(y - check_y) < 1e-4, f"Evaluated y are not equal. Evaluated: {y}. Expected: {check_y}"

        if analytical:
            var_y1 = self.stderr_for_x(x, return_var=True, analytical=True)
            y2 = other_fn.evaluate_y_for_x(x)
            var_y2 = other_fn.stderr_for_x(x, return_var=True, analytical=True)

            offset = y2 - y
            stderr = np.sqrt(var_y2 + var_y1)
            cdf = 1 - stats.norm.cdf(0.0, loc=offset, scale=stderr)
        else:
            warnings.warn("Empirical stderr for GP Regression is expensive! Returning Nones")
            return None, None, None

        return offset.item(), stderr.item(), cdf.item()

    def reformat_input(self, input):
        """
        Format's input that is recognizable by GPR model.
        """
        if type(input) in [int, float]:
            return np.array([[input]])
        elif len(input.shape) == 1:
            return input.reshape(-1, 1)
        elif len(input.shape) == 2:
            return input
        else:
            raise NotImplementedError(f"Unknown type: {type(input)} of y: {input}")

    def sample_f(self):
        """
        Runs MCMC to sample posterior functions.
        """
        # add priors to the hyperparameters.
        self.model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        self.model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        if self.mean_function is not None:
            self.model.mean_function.A.prior = tfd.Normal(f64(0.0), f64(10.0))
            self.model.mean_function.b.prior = tfd.Normal(f64(0.0), f64(10.0))

        # sample from the posterior using HMC (required to estimate epistemic uncertainty)
        num_burnin_steps = ci_niter(300)
        num_samples = ci_niter(self.num_samples)

        # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
        self.hmc_helper = gpflow.optimizers.SamplingHelper(
            self.model.log_posterior_density, self.model.trainable_parameters
        )

        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
        )
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
        )

        @tf.function
        def run_chain_fn():
            return tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=self.hmc_helper.current_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )

        self.samples, traces = run_chain_fn()


    def save(self, path):
        assert Path(path).parent.exists(), f"{path} doesn't exist"
        self.model.predict_f_compiled = tf.function(
                self.model.predict_f, input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)]
            )

        tf.saved_model.save(self.model, path)

    def load(self, path):
        self.using_saved_model = True
        self.model = tf.saved_model.load(str(Path(path).resolve()))
        self.model.predict_f = self.model.predict_f_compiled


def bootstrap_series(series, num_bootstrap_samples=100, seed=0, mode="mean"):
    """
    Implements bootstrapping of a time series to sample underlying population of means.

    Args:
        series (np.array):  of size (M, T) where M is the number of series in to sample from
        num_bootstrap_samples (int): number of samples to return
        seed (int): seed for random number generator

    Returns:
        (np.array): of size  (num_bootstrap_samples, T)
    """
    # series.shape = (N, T)
    rng = np.random.default_rng(seed)
    subset_size = series.shape[0]

    means = []
    for k in range(num_bootstrap_samples):
        if mode == "mean":
            ufunc = np.mean
        elif mode == "median":
            ufunc = np.median

        means.append(ufunc(rng.choice(series, subset_size, replace=True, axis=0), axis=0))
    return np.array(means)

def ewma(data, window):
    """
    Implements moving average for `window` size.

    Args:
        data (np.array):
        window (int):

    Returns:
        ()
    """
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def bootstrap(pd_series, num_bootstrap_samples=1000):
    """
    Bootstraps from the given data to estimate distribution

    Args:
        pd_series (pd.Series):

    Returns:
        (list)
    """
    rng = np.random.RandomState(1)

    out = []
    for _ in range(num_bootstrap_samples):
        estimand = pd_series.sample(len(pd_series), replace=True, random_state=rng).to_numpy()
        estimate = np.mean(estimand)
        out.append(estimate)
    return out
