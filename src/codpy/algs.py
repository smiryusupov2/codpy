import pandas as pd
from codpydll import *

from codpy.data_conversion import get_matrix
from codpy.data_processing import lexicographical_permutation

from .core import _kernel_helper2


class alg:
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
            "set_codpykernel": _kernel_helper2(
                kernel=kernel_fun,
                map=map,
                polynomial_order=polynomial_order,
                regularization=regularization,
            )
        }
        if rescale == True or _requires_rescale(map_name=map):
            params["rescale"] = True
            params["rescalekernel"] = rescale_params
            if verbose:
                warnings.warn(
                    "Rescaling is set to True as it is required for the chosen map."
                )
            kernel_interface.init(x, x, x, **params)
        else:
            params["rescale"] = rescale
            kernel_interface.init(**params)
        Nx, Dx = np.shape(x)
        Nx, Df = np.shape(fx)
        Ny = len(probas)
        fy, y, permutation = fun_permutation(fx, x)
        out = np.concatenate((y, fy), axis=1)
        quantile = np.array(np.arange(start=0.5 / Nx, stop=1.0, step=1.0 / Nx)).reshape(
            Nx, 1
        )
        out = op.projection(
            x=quantile,
            y=probas,
            z=probas,
            fx=out,
            kernel_fun=kernel_fun,
            rescale=rescale,
        )
        return out[:, 0:Dx], out[:, Dx:], permutation

    def Pi(x, z, fz=[], nmax=10, rescale=False, **kwargs):
        # print('######','Pi','######')
        kernel_interface.init(**kwargs)
        if rescale:
            kernel_interface.rescale(x, z)
        out = cd.alg.Pi(x=x, y=x, z=z, fz=fz, nmax=nmax)
        return out

    def HybridGreedyNystroem(
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


    def add(Knm, Knm_inv, x, y):
        # import codpy.core
        # codpy.core.set_verbose(True)
        return cd.alg.add(Knm, Knm_inv, x, y)


if __name__ == "__main__":
    from include_all import *

    x, fx = np.random.rand(10, 2), np.random.rand(10, 3)
    lalg.prod(x, x)
    kernel_interface.rescale(x)
    Knm = op.Knm(x=x, y=x)
    Knm_inv = lalg.cholesky(x=Knm, eps=1e-2)
    Knm_inv = lalg.lstsq(A=Knm)

    print(op.Knm_inv(x=x, y=x, fx=op.Knm(x=x, y=x)))
    alg.HybridGreedyNystroem(x=x, fx=fx)
    pass