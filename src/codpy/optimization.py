import sys
import numpy as np
import scipy
from scipy import optimize
from codpy.core import get_matrix


def screen_optimize(function, distribution, n=1000, maximize=True):
    if n == 0:
        return None
    samples = distribution(n)
    y = function(samples).flatten()

    if maximize:
        y_extrem_idx = y.argmax()
    else:
        y_extrem_idx = y.argmin()

    extremum = y[y_extrem_idx], samples[y_extrem_idx]

    return extremum


def continuous_optimizer(
    function,
    distribution,
    n=1000,
    n_iter=5,
    contracting_factor=0.3,
    maximize=True,
    include=None,
    verbose=False,
    **kwargs
):
    if maximize == False:
        return continuous_optimizer(
            lambda x: -function(x),
            distribution,
            n,
            n_iter,
            contracting_factor,
            True,
            include=include,
            verbose=verbose,
            **kwargs
        )

    class contract_distrib:
        def __init__(self, contracting_factor, mean):
            self.contracting_factor = contracting_factor
            self.mean = mean

        def __call__(self, n, *args, **kwds):
            samples = distribution(n)
            if self.mean is not None:
                samples = (
                    samples - samples.mean()
                ) * self.contracting_factor + self.mean
                samples = distribution.support(samples)
            return samples

    mean = None
    cf = 1.0
    extremum_y, extremum_x = -sys.float_info.max, None
    if include is not None:
        values = function(include)
        extremum_y, extremum_x = values.max(), include[values.argmax()]
    if verbose==True:
        print(f"Initial point value: {extremum_y}")
    for i in range(min(n, n_iter)):
        temp_y, temp_x = screen_optimize(
            function, contract_distrib(cf, mean), n, maximize
        )
        if temp_y > extremum_y:
            extremum_y = temp_y
            extremum_x = temp_x
        mean = extremum_x
        cf *= contracting_factor

    if verbose==True:
        print(f"Final point value: {extremum_y}")
    return extremum_y, extremum_x


def grad_optimizer(
    function,
    distribution,
    n=1000,
    n_iter=5,
    contracting_factor=0.3,
    maximize=True,
    **kwargs
):
    extremum_y, extremum_x = continuous_optimizer(
        function, distribution, n, n_iter, contracting_factor, maximize, **kwargs
    )
    if hasattr(function, "grad"):
        extremum_x = get_matrix(extremum_x).T
        grad_ = function.grad(extremum_x).reshape(extremum_x.shape)
        grad_ /= np.fabs(grad_).max() + 1e-9

        # grad_ = distribution.support(extremum_x + grad_) - extremum_x
        def f(x):
            x = max(0.0, min(x, 1e-2))
            interpolated_thetas = distribution.support(extremum_x + grad_ * x)
            out = -function(interpolated_thetas)
            return out

        delta = (f(1e-9) - f(0.0)) / 1e-9
        xmin, fval, iter, funcalls = optimize.brent(
            f, brack=(0.0, 1.0), maxiter=5, full_output=True
        )
        if -fval > extremum_y + 1e-9:
            xmin = max(0.0, min(xmin, 1e-2))
            # xmin = max(0.,min(xmin,1.))
            extremum_x = distribution.support(extremum_x + grad_ * xmin)
            # xmin = max(0.,min(xmin,1.))
            extremum_y = function(extremum_x)

    return extremum_y, get_matrix(extremum_x).T
