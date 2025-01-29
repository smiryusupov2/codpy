import math
import warnings
import functools
import numpy as np
from codpydll import *
from sklearn.cluster import KMeans, MiniBatchKMeans

import codpy.algs
from codpy.core import _requires_rescale,KerInterface,kernel_setter,get_matrix
from codpy.data_conversion import get_matrix
from codpy.selection import column_selector
from codpy.kernel import Kernel
from codpy.lalg import LAlg



def kernel_conditional_density_estimator(x, y, kernel_x=None,kernel_y=None,**kwargs):
    """
    Estimate the conditional density of 'Y' given 'X' using Nadaraya-Watson kernel conditional density estimator.
    that is the matrix of conditional probabilities p(y|x) for each x in X and y in Y.

    This function calculates the conditional density of values based on a joint distribution ('X', 'Y').
    It uses KDE method for the estimation.

    Args:
        X (array-like): Observed data for 'x' in the joint distribution with 'y'.
        Y (array-like): Observed data for 'y' in the joint distribution with 'x'.

    pre-requisite:
        A kernel must be loaded and rescaled to the data X AND Y

    Returns:
        array-like: The estimated conditional density of 'Y' given 'X', that is a stochastic matrix.

    Example:
        Define joint distribution data for 'x' and 'y'

        >>> X = np.array([...])
        >>> Y = np.array([...])

        Compute the conditional density

        >>> conditional_density = kernel_conditional_density_estimator(X, Y)
    """
    # given a joint distribution (X, Y), return the density Y | X using the Nadaraya-Watson estimate
    if kernel_x is None:
        kernel_x = Kernel(x=x,**kwargs)
    kernel_x.set_kernel_ptr()
    marginal_x = kernel_x.knm(x=x, y=kernel_x.get_x())

    if kernel_y is None:
        kernel_y = Kernel(x=y,**kwargs)
    kernel_y.set_kernel_ptr()
    marginal_y = kernel_y.knm(x=y, y=kernel_y.get_x())
    out = np.zeros([x.shape[0], y.shape[0]])
    def helper(i,j):
        out[i,j] = (marginal_x[i] * marginal_y[j]).sum() / (marginal_x[i].sum())
    [helper(i,j) for i in range(x.shape[0]) for j in range(y.shape[0])]
    return out

def rejection_sampling(proposed_sample, probas, acceptance_ratio=0.0):
    """
    Perform rejection sampling on a set of proposed samples.

    This function implements the rejection sampling algorithm, a technique in Monte Carlo methods.
    It evaluates each proposed sample against an acceptance criterion based on the sample's probability and
    an acceptance ratio. Samples are accepted with a probability proportional to their probability in the
    target distribution.

    Args:
        proposed_sample (array-like): An array of proposed samples to be evaluated.
        probas (array-like): An array of probabilities corresponding to each proposed sample.
        acceptance_ratio (float, optional): A threshold ratio for accepting samples. This value can be used
                                            to control the acceptance rate. Default is 0.

    Returns:
        list: A list of samples that are accepted based on the rejection sampling criterion.

    Example:
        Proposed samples and their probabilities

        >>> proposed_samples = np.array([...])
        >>> probabilities = np.array([...])

        Perform rejection sampling

        >>> accepted_samples = rejection_sampling(proposed_samples, probabilities)

        Note:
        The function assumes that the proposed samples and their probabilities are of the same length.
    """
    samples = []
    for n in range(proposed_sample.shape[0]):
        acceptance_ratio = max(acceptance_ratio, probas[n])
        if np.random.uniform(0, acceptance_ratio) < probas[n]:
            samples.append(proposed_sample[n])
    return samples


@functools.cache
def get_normals(N, D, nmax=10,kernel_ptr=None):
    if kernel_ptr is not None:
        KerInterface.set_kernel_ptr(kernel_ptr)
    else:
        kernel_setter("maternnorm", "standardmean", 0, 1e-9)()
    out = cd.alg.get_normals(N=N, D=D, nmax=nmax)
    # mean,var = np.mean(out,axis=0),np.var(out,axis=0)
    return out


@functools.cache
def get_uniforms(N, D, nmax=10,kernel_ptr=None):
    """
    Generate uniformly distributed random samples from normally distributed samples.

    This function first generates random samples from a normal distribution and then
    transforms them into a uniform distribution using the error function (erf). It's
    based on the probability integral transform where the Gaussian CDF is used for this conversion.

    Args:
        N (int): The number of samples to generate.
        D (int): The dimensionality of each sample.
        nmax (int): .

    Returns:
        ndarray: An array of shape (N, D) containing uniformly distributed random samples.

    Example:
        Generate 100 samples with 2 dimensions

        >>> uniform_samples = get_uniforms(100, 2)
    """
    out = get_normals(N, D, nmax=nmax,kernel_ptr=kernel_ptr)
    out = np.vectorize(math.erf)(out) / 2.0 + 0.5
    return out


def get_uniforms_like(x, kernel_ptr=None,**kwargs):
    return get_uniforms(N=x.shape[0], D=x.shape[1],kernel_ptr=kernel_ptr)


def get_normals_like(x, kernel_ptr=None,**kwargs):
    return get_normals(N=x.shape[0], D=x.shape[1],kernel_ptr=kernel_ptr)


def get_random_normals_like(x, **kwargs):
    return np.random.normal(size=x.shape)


def get_random_uniforms_like(x, **kwargs):
    return np.random.uniform(size=x.shape)


def match(x, N, **kwargs):
    if N >= x.shape[0]:
        return x
    out = cd.alg.match(get_matrix(x), N)
    return out