import numpy as np
import scipy


def optimize(function, distribution, n=1000, maximize=True):
    samples = distribution(n)

    y = function(samples)

    assert y.ndim == 1, f"Excpeting 1 dimensionnal array for {function.__name__}"

    if maximize:
        y_extrem_idx = y.argmax()
    else:
        y_extrem_idx = y.argmin()

    extremum = y[y_extrem_idx], samples[y_extrem_idx]

    return extremum


def continuous_optimizer(
    function, distribution, n=1000, n_iter=5, contracting_factor=0.5, maximize=True
):
    class contract_distrib:
        def __init__(self, contracting_factor, mean):
            self.contracting_factor = contracting_factor
            self.mean = mean

        def __call__(self, n, *args, **kwds):
            samples = distribution(n)
            if self.mean is not None:
                return (samples - samples.mean()) * self.contracting_factor + self.mean
            return samples

    mean = None
    cf = 1.0
    for i in range(n_iter):
        extremum_y, extremum_x = optimize(
            function, contract_distrib(cf, mean), n, maximize
        )
        mean = extremum_x
        cf *= contracting_factor

    return extremum_y, extremum_x
