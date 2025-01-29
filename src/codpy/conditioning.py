import math
import warnings
import functools
import numpy as np
import abc

import codpy.algs
from codpy.core import _requires_rescale, KerInterface, kernel_setter, get_matrix
from codpy.data_conversion import get_matrix
from codpy.selection import column_selector
from codpy.kernel import Kernel
from codpy.lalg import LAlg
from codpy.plot_utils import multi_plot
import pandas as pd


def NWkernel_density_estimator(x, y, kernel=None, **kwargs):
    """
     Estimate the kernel density of a distribution x at points y.

     This function implements a kernel density estimator (KDE), a non-parametric method to estimate
     the probability density function of a random variable. It evaluates the density estimate based on
     two input distributions using a specified kernel function.

    Args:
         **kwargs: Arbitrary keyword arguments, including:
             x (array-like): The first input distribution for which the density estimate is to be computed.
             y (array-like): The second input distribution used in the density estimation process.
             kernel (optional): The kernel function to be used for density estimation. This can be specified
                             as part of the kwargs. If not specified, a default kernel is used.

     Returns:
         array-like: The estimated density values based on the kernel density estimation.

     Example:
         Two sample distributions

         >>> x = np.array([...])
         >>> y = np.array([...])

         Compute the kernel density estimation

         >>> density = kernel_density_estimator(x=x, y=y)
    """
    if kernel is None:
        kernel = Kernel(x=np.concatenate((x, y), axis=0), **kwargs)
    kernel.set_kernel_ptr()
    out = kernel.knm(x, y)
    out /= get_matrix(out.sum(axis=1))
    return out


class Conditionner:

    def __init__(self, x: np.ndarray, y: np.ndarray, **kwargs):
        assert x.shape[0] == y.shape[0]
        self.x, self.y = x, y

    def __call__(self, x: np.ndarray, **kwargs):
        assert x.shape[1] == self.x.shape[1]
        return self.expectation(x)

    def expectation(self, x, **kwargs):
        raise NotImplementedError

    def sample(self, x, n):
        raise NotImplementedError

    def var(self, x):
        raise NotImplementedError

    def density(self, y, x):
        raise NotImplementedError


class NadarayaWatsonKernel(Kernel):
    def __init__(self, x, y, **kwargs):
        """
        Base class to handle Nadaraya-Watson kernel conditional estimators of the law y | x.
        """
        super().__init__(x=x, fx=y, **kwargs)

    def __call__(self, z, **kwargs):
        """
        Return the Nadaraya-Watson kernel conditional mean estimator at each points z.
        """
        probas = NWkernel_density_estimator(x=z, y=self.get_x(), kernel=self, **kwargs)
        out = LAlg.prod(probas, self.get_fx())
        return out

    def var(self, z, **kwargs):
        """
        Return the Nadaraya-Watson kernel conditional var estimator at each points z.
        """
        probas = NWkernel_density_estimator(x=z, y=self.get_x(), kernel=self, **kwargs)
        expectations = LAlg.prod(probas, self.get_fx())
        vars = np.zeros([z.shape[0], self.get_fx().shape[1], self.get_fx().shape[1]])

        def helper(i):
            temp = self.get_fx() - expectations[i]
            proba = get_matrix(probas[i])
            temp = temp * np.sqrt(proba)  # not sure here
            temp = temp.T @ temp
            vars[i, :] = temp.mean(axis=0)

        [helper(i) for i in range(z.shape[0])]

        return vars


class ConditionerKernel(Kernel):
    def __init__(self, x, y, latent_x=None, latent_y=None, **kwargs):
        """
        Base class to handle kernel conditional estimators of the law y | x using optimal transport
        """
        super().__init__(x=get_matrix(x), fx=get_matrix(y), **kwargs)
        if latent_x is None:
            self.latentd_x = lambda N: np.random.normal(size=[N, self.get_x().shape[1]])
        else:
            self.latentd_x = latent_x
        if latent_y is None:
            self.latentd_y = lambda N: np.random.normal(
                size=[N, self.get_fx().shape[1]]
            )
        else:
            self.latent_y = latent_y
        self.latent_x = self.latentd_x(self.get_x().shape[0])
        self.latent_y = self.latentd_y(self.get_fx().shape[0])

        latent_xy = np.concatenate((self.latent_x, self.latent_y), axis=1)
        xy = np.concatenate((x, y), axis=1)
        self.map_xy = Kernel(x=latent_xy).map(y=xy, **kwargs)
        self.map_xy_inv = Kernel(
            x=self.map_xy.get_fx(), fx=self.map_xy.get_x(), **kwargs
        )
        self.map_x = Kernel(x=self.latent_x).map(y=self.get_x(), **kwargs)
        self.map_x_inv = Kernel(x=self.map_x.get_fx(), fx=self.map_x.get_x(), **kwargs)

    def __call__(self, z, **kwargs):
        """
        Return the kernel conditional mean estimator at each points z.
        """
        probas = NWkernel_density_estimator(x=z, y=self.get_x(), kernel=self, **kwargs)
        out = LAlg.prod(probas, self.get_fx())
        return out

    def samples(self, z, N, **kwargs):
        """
        Return N sampling for each z of the estimated law y | z.
        """
        probas = NWkernel_density_estimator(x=z, y=self.get_x(), kernel=self, **kwargs)
        expectations = LAlg.prod(probas, self.get_fx())
        vars = np.zeros([z.shape[0], self.get_fx().shape[1], self.get_fx().shape[1]])

        def helper(i):
            temp = self.get_fx() - expectations[i]
            proba = get_matrix(probas[i])
            temp = temp * np.sqrt(proba)  # not sure here
            temp = temp.T @ temp
            vars[i, :] = temp.mean(axis=0)

        [helper(i) for i in range(z.shape[0])]

        return vars
