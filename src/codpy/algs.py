import warnings

import numpy as np
from codpydll import *

import codpy.core
from codpy.core import KerInterface, KerOp, _requires_rescale
from codpy.data_conversion import get_matrix
from codpy.data_processing import lexicographical_permutation
from codpy.lalg import LAlg


class Alg:
    def _iso_probas_projection(
        x,
        fx,
        probas,
        fun_permutation=lexicographical_permutation,
        kernel_fun=None,
        map=None,
        polynomial_order=2,
        regularization=1e-8,
        rescale=False,
        rescale_params: dict = {"max": 1000, "seed": 42},
        verbose=False,
        **kwargs,
    ):
        # print('######','iso_probas_projection','######')
        params = {
            "set_codpykernel": codpy.core._kernel_helper2(
                kernel=kernel_fun,
                map=map,
                polynomial_order=polynomial_order,
                regularization=regularization,
            )
        }
        if rescale or _requires_rescale(map_name=map):
            params["rescale"] = True
            params["rescalekernel"] = rescale_params
            if verbose:
                warnings.warn(
                    "Rescaling is set to True as it is required for the chosen map."
                )
            KerInterface.init(x, x, x, **params)
        else:
            params["rescale"] = rescale
            KerInterface.init(**params)
        Nx, Dx = np.shape(x)
        Nx, Df = np.shape(fx)
        Ny = len(probas)
        fy, y, permutation = fun_permutation(fx, x)
        out = np.concatenate((y, fy), axis=1)
        quantile = np.array(np.arange(start=0.5 / Nx, stop=1.0, step=1.0 / Nx)).reshape(
            Nx, 1
        )
        out = KerOp.projection(
            x=quantile,
            y=probas,
            z=probas,
            fx=out,
            kernel_fun=kernel_fun,
            rescale=rescale,
        )
        return out[:, 0:Dx], out[:, Dx:], permutation

    def pi(x, y, z=None, fz=None, nmax=5, kernel_ptr = None,order=None,reg=1e-8,**kwargs):
        # print('######','Pi','######')
        from codpy.kernel import KernelClassifier
        if kernel_ptr is not None:
            KerInterface.set_kernel_ptr(kernel_ptr,order,reg)

        out = cd.alg.Pi(x=x, y=y, nmax=nmax)
        if z is not None and fz is not None:
            out = LAlg.prod(KernelClassifier(x=x, fx=out, **kwargs)(z),fz)
        return out

    def hybrid_greedy_nystroem(
        x,
        fx,
        tol=1e-5,
        N=0,
        n_batch=10,
        error_type="classifier",
        start_indices=[],
        **kwargs,
    ):
        # NumPy arrays input
        x, fx = get_matrix(x), get_matrix(fx)
        cn, indices = cd.alg.HybridGreedyNystroem(
            x, fx, start_indices, tol, N, n_batch, error_type
        )
        return cn, indices

    def balanced_clustering(D):
        return cd.alg.balanced_clustering(D)

    def two_balanced_clustering(DX, DY, C):
        labels1, labels2 = cd.alg.two_balanced_clustering(DX, DY, C)
        return np.array(labels1), np.array(labels2)

    def add(knm, knm_inv, x, y, kernel_ptr = None,order=None,reg=1e-8):
        # import codpy.core
        # codpy.core.set_verbose(True)
        if kernel_ptr is not None:
            KerInterface.set_kernel_ptr(kernel_ptr,order,reg)
        out = cd.alg.add(knm, knm_inv, x, y)
        return out

    def greedy_algorithm(
        x,
        N,
        start_indices=set(),
        **kwargs,
    ):
        # NumPy arrays input
        x = get_matrix(x)
        out = cd.tools.greedy_algorithm(get_matrix(x), N, start_indices)
        return out

    def probas_projection(probs, axis=1,**kwargs):
        if axis==1:
            out = probs / probs.sum(axis=1).reshape(-1, 1)
        else:
            out = Alg.probas_projection(probs.T,axis=1).T
        return out

    def proportional_fitting(probs, iter=100, axis=0,**kwargs):
        if axis==1:
            out = cd.alg.proportional_fitting(probs.T, iter).T
            out = out / out.sum(axis=1).reshape(-1, 1)
        else:
            out = cd.alg.proportional_fitting(probs, iter)
        return out


if __name__ == "__main__":
    from include_all import *

    x, fx = np.random.rand(10, 2), np.random.rand(10, 3)
    LAlg.prod(x, x)
    KerInterface.rescale(x)
    Knm = KerOp.knm(x=x, y=x)
    Knm_inv = LAlg.cholesky_inverse(x=Knm, eps=1e-2)
    Knm_inv = LAlg.lstsq(A=Knm)

    print(KerOp.knm_inv(x=x, y=x, fx=KerOp.knm(x=x, y=x)))
    alg.HybridGreedyNystroem(x=x, fx=fx)
    pass
