import math
import warnings
import functools
import numpy as np
import abc

import codpy.algs
import itertools
from codpy.core import _requires_rescale, KerInterface, kernel_setter, get_matrix
from codpy.data_conversion import get_matrix
from codpy.selection import column_selector
from codpy.kernel import Kernel, KernelClassifier
from codpy.sampling import rejection_sampling
from codpy.lalg import LAlg
from codpy.plot_utils import multi_plot
from codpy.utils import cartesian_outer_product
import codpy.algs
import pandas as pd


class Conditionner:
    """
    Base class to handle conditioned probability.

    This class is intended to standardized conditioned probability handling exhibiting the basic functionalities to implement.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        The constructor takes both distributions x and y in order to model $y|x$.
        """
        x, y = get_matrix(x), get_matrix(y)
        assert x.shape[0] == y.shape[0]
        self.x, self.y = x, y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __call__(self, x: np.ndarray, **kwargs):
        """
        A shortcut to the conditional expectation $\mathbb{E}(y|x)$.
        """
        assert x.shape[1] == self.x.shape[1]
        return self.expectation(x)

    def expectation(self, x, **kwargs):
        """
        Return the estimator of the conditional expectation $\mathbb{E}(y|x)$
        The output is expected to have size $(x.shape[0],y.shape[1])$.
        """
        probas = self.joint_density(x=x, y=self.y, **kwargs)
        probas /= get_matrix(probas.sum(axis=1))
        out = LAlg.prod(probas, self.y)
        return out

    def sample(self, x, n, **kwargs):
        """
        Return $n$ i i d samples of the conditional law $y|x$.
        The output is expected to be a three-dimensional array having size $(n,x.shape[0],y.shape[1])$
        """
        raise NotImplementedError

    def var(self, x, **kwargs):
        """
        Return the estimator of the conditional variance $\mathbb{E}(y|x)$
        The output is expected to have size $(x.shape[0],y.shape[1]**2)$.
        """
        raise NotImplementedError

    def density(self, x, **kwargs):
        """
        Return the estimation of the density of the law $p(x^i)$
        The output is expected to have size $x.shape[0]$.
        """
        raise NotImplementedError

    def joint_density(self, x, y, **kwargs):
        """
        Return the estimation of the density of the law $p(x^i,y^j)$
        The output is expected to have size $(x.shape[0],y.shape[0])$.
        """
        raise NotImplementedError


class NadarayaWatsonKernel(Conditionner):
    class KDE:
        def __init__(self, k):
            self.kernel = k

        def __call__(self, x):
            out = self.kernel.knm(x=self.kernel.get_x(), y=x).sum(axis=0)
            return out

    class joint_KDE:
        def __init__(self, kx, ky):
            self.kernelx = kx
            self.kernely = ky

        def __call__(self, x, y):
            out = np.zeros([x.shape[0], y.shape[0]])
            kxx = self.kernelx.knm(x=x, y=self.kernelx.get_x())
            kyy = self.kernely.knm(x=self.kernely.get_x(), y=y)
            out = LAlg.prod(kxx, kyy)
            return out

    def __init__(self, x, y, **kwargs):
        """
        Base class to handle Nadaraya-Watson kernel conditional estimators of the law y | x.
        """
        super().__init__(x=x, y=y, **kwargs)
        self.density_x = NadarayaWatsonKernel.KDE(Kernel(x=x))
        self.density_xy = NadarayaWatsonKernel.joint_KDE(Kernel(x=x), Kernel(x=y))
        self.expectation_kernel = None

    def var(self, z, **kwargs):
        """
        Return the Nadaraya-Watson kernel conditional var estimator at each points z.
        """
        probas = self.density_x(z)
        probas /= probas.sum()
        expectations = LAlg.prod(probas, self.y)
        vars = np.zeros([self.x.shape[0], self.y.shape[1], self.y.shape[1]])

        y_norm = self.y - self.expectation(self.x)  # n,D
        for i in range(self.x.shape[0]):
            vars[i, :] = y_norm[i].T @ y_norm[i]  # D,1 x 1,D => D,D

        var_flat = vars.reshape((vars.shape[0], vars.shape[1] * vars.shape[2]))

        var_estim = NadarayaWatsonKernel(self.x, var_flat)
        out = var_estim(z).reshape(
            [z.shape[0], vars.shape[1], vars.shape[2]]
        )  # z.shape[0], D, D
        return out

    def sample(self, x, n, **kwargs):
        out = np.zeros([x.shape[0], n, self.y.shape[1]])
        density_xy = LAlg.prod(
            self.density_xy.kernely.get_knm(),
            self.density_xy.kernelx.knm(x=self.x, y=x),
        ).T

        def helper(i):
            temp = rejection_sampling(self.y, density_xy[i], size=[n])
            out[i, :] = temp

        [helper(i) for i in range(x.shape[0])]
        return out

    def get_expectation_kernel(self,**kwargs):
        # esp(y|x) - diff de proba de transition
        class expectation_kernel:
            def __init__(
                self, call_back,**kwargs
            ):
                super().__init__(**kwargs)
                self.call_back = call_back

            def __call__(self,z,**kwargs):
                density_x = self.call_back.density_x.kernel.knm(z,self.call_back.density_x.kernel.get_x())
                density_x /= density_x.sum(axis=1)[:,None]
                fx = self.call_back.get_y()
                return density_x@fx
        if self.expectation_kernel is None and self.x is not None:
            self.expectation_kernel = expectation_kernel(self, **kwargs)
        return self.expectation_kernel

    def expectation(self, x=None, **kwargs):
        """
        Return the estimator of the conditional expectation $\mathbb{E}(f(y)|x)$
        The output is expected to have size $(x.shape[0],y.shape[1])$.
        """
        return self.get_expectation_kernel()(x)


    def density(self, x, **kwargs):
        """
        Return the estimation of the density of the law $p(x)$
        The output is expected to have size $(y.shape[0],x.shape[0])$.
        """
        return self.density_x(x)

    def joint_density(self, x, y, **kwargs):
        """
        Return the estimation of the density of the law $p(x,y)$
        The output is expected to have size $(x.shape[0],y.shape[0])$.
        """
        return self.density_xy(x, y)

    def dnm(self, x, y, **kwargs):
        """
        Return the kernel induced distance on the x-space $d(x,y)=k(x,x)+k(y,y)-2k(x,y)$
        The output is expected to have size $(x.shape[0],y.shape[0])$.
        """
        return self.density_x.kernel.dnm(x, y)


class ConditionerKernel(Conditionner):
    def __init__(
        self, x, y, latent_generator_x=None, latent_generator_y=None, **kwargs
    ):
        """
        Base class to handle kernel conditional estimators of the law y | x using optimal transport
        """
        x, y = get_matrix(x), get_matrix(y)
        super().__init__(x=x, y=y, **kwargs)
        xy = np.concatenate([x, y], axis=1)
        self.cut_ = x.shape[1]
        self.latent_generator_xy = lambda n: np.random.normal(
            size=[n, x.shape[1] + y.shape[1]]
        )
        self.latent_generator_x = lambda n: np.random.normal(size=[n, x.shape[1]])
        self.latent_generator_y = lambda n: np.random.normal(size=[n, y.shape[1]])
        self.latent_x, self.latent_y = self.latent_generator_x(
            x.shape[0]
        ), self.latent_generator_y(x.shape[0])
        self.latent_xy = np.concatenate([self.latent_x, self.latent_y], axis=1)

        self.map_xy_inv = Kernel(x=self.latent_xy, **kwargs).map(y=xy)
        self.map_xy = Kernel(
            x=self.map_xy_inv.get_fx(), fx=self.map_xy_inv.get_x(), **kwargs
        )
        latent_x = self.map_xy_inv.get_x()[:, : x.shape[1]]
        im_x = self.map_xy_inv.get_fx()[:, : x.shape[1]]
        self.map_x = Kernel(x=im_x, fx=latent_x, **kwargs)
        self.pi = None
        self.expectation_kernel = None
        self.var_kernel = None

    def get_pi(self):
        # Estime pi(y|x) avec y et x donnés comme exemple, matrice de transition, proba d'avoir couple y_j sachant x_j etc
        if self.pi is None and self.x is not None:
            self.pi = codpy.algs.Alg.pi(
                self.get_x(), self.get_y(), self.map_x.get_kernel()
            )
            self.pi = Kernel(x=self.get_x(), fx=self.pi.copy(), order=0)
        return self.pi

    def var(self, z, **kwargs):

        out = self.get_var_kernel()(z)
        return out

    def get_var_kernel(self,**kwargs):
        class var_kernel(Kernel):
            def __call__(self,z,**kwargs):
                return super().__call__(z, **kwargs).reshape(z.shape[0],self.get_fx().shape[1],self.get_fx().shape[1])
            
        if self.var_kernel is None and self.x is not None:
            vars = np.zeros([self.x.shape[0], self.y.shape[1], self.y.shape[1]])
            y_norm = self.y - self.get_expectation_kernel(**kwargs)(self.x)
            def helper(i):
                vars[i, :] = y_norm[i].T @ y_norm[i]
            [helper(i) for i in range(self.x.shape[0])]
            self.var_kernel = var_kernel(x=self.x, fx=vars.reshape(vars.shape[0],vars.shape[1]*vars.shape[2]), **kwargs)
        return self.var_kernel


    def get_expectation_kernel(self,**kwargs):
        # esp(y|x) - diff de proba de transition
        class expectation_kernel(Kernel):
            def __init__(
                self, call_back,**kwargs
            ):
                super().__init__(**kwargs)
                self.call_back = call_back

            def __call__(self,z,**kwargs):
                mapped_z = self.call_back.map_x(z, **kwargs)
                expectation_y = super().__call__(self.call_back.map_x(z, **kwargs))
                return self.call_back.map_xy_inv(np.concatenate([mapped_z,expectation_y],axis=1))[:,z.shape[1]:]
        if self.expectation_kernel is None and self.x is not None:
            self.expectation_kernel = expectation_kernel(call_back=self,x=self.latent_x, fx=self.latent_y, **kwargs)
        return self.expectation_kernel

    def expectation(self, x=None, **kwargs):
        """
        Return the estimator of the conditional expectation $\mathbb{E}(f(y)|x)$
        The output is expected to have size $(x.shape[0],y.shape[1])$.
        """
        return self.get_expectation_kernel()(x)

    def sample(self, x, n, **kwargs):
        """
        Return N sampling for each z of the estimated law y | z.
        output is of size (x.shape[0],n,y.shape[1])
        """
        latent_x = self.map_x(x, **kwargs)
        latent_y = self.latent_generator_y(n)
        latent_xy = cartesian_outer_product(latent_x, latent_y).reshape(
            latent_x.shape[0] * latent_y.shape[0], self.map_xy.get_fx().shape[1]
        )
        mapped = self.map_xy_inv(latent_xy).reshape(
            latent_x.shape[0], latent_y.shape[0], self.map_xy_inv.get_fx().shape[1]
        )
        mapped = mapped[:, :, self.cut_ :]
        return mapped

        # return out+self.meany

    def density(self, x, **kwargs):
        """
        Return the estimation of the density of the law $p(x)$
        The output is expected to have size $(y.shape[0],x.shape[0])$.
        """
        return self.map_x.density(x)

    def joint_density(self, x, y, **kwargs):
        """
        Return the estimation of the density of the law $p(x,y)$
        The output is expected to have size $(x.shape[0],y.shape[0])$.
        """
        return self.map_xy.density(np.concatenate([x,y],axis=0))
