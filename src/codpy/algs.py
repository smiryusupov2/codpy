import warnings

import numpy as np
from codpydll import *
import faiss
faiss.omp_set_num_threads(10)
import scipy.sparse as sp
from typing import Tuple

import codpy.core
import time
from scipy import optimize
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
            "set_codpykernel": codpy.core.set_kernel(
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

    def pi(
        x, y, z=None, fz=None, nmax=5, kernel_ptr=None, order=None, reg=1e-8, **kwargs
    ):
        # print('######','Pi','######')
        from codpy.kernel import KernelClassifier

        if kernel_ptr is not None:
            KerInterface.set_kernel_ptr(kernel_ptr, order, reg)

        out = cd.alg.Pi(x=x, y=y, nmax=nmax)
        if z is not None and fz is not None:
            out = LAlg.prod(KernelClassifier(x=x, fx=out, **kwargs)(z), fz)
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

    def add(knm, knm_inv, x, y, kernel_ptr=None, order=None, reg=1e-8):
        # import codpy.core
        # codpy.core.set_verbose(True)
        if kernel_ptr is not None:
            KerInterface.set_kernel_ptr(kernel_ptr, order, reg)
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

    def probas_projection(probs, axis=1, **kwargs):
        if axis == 1:
            out = probs / probs.sum(axis=1).reshape(-1, 1)
        else:
            out = Alg.probas_projection(probs.T, axis=1).T
        return out

    def proportional_fitting(probs, iter=100, axis=0, **kwargs):
        if axis == 1:
            out = cd.alg.proportional_fitting(probs.T, iter).T
            out = out / out.sum(axis=1).reshape(-1, 1)
        else:
            out = cd.alg.proportional_fitting(probs, iter)
        return out
    def conjugate_gradient_descent(A,
                        b,
                        steps=100,
                        alpha=.99,
                        verbose=False,
                        constraints=None,
                        dot_product=lambda A,x: A@x,
                        threshold = 1e-4,
                        **kwargs
                        ):
        timer = time.perf_counter()
        yk = b.copy()
        rk = alpha*yk - dot_product(A,yk)
        pk = rk
        fstart=None

        for n in range(steps):
            alpha_k = (rk*rk).sum() 
            alpha_k /= (pk * (dot_product(A,pk))).sum()
            rk1 = rk - alpha_k * dot_product(A,pk)
            yk1 = yk+alpha_k*pk
            if constraints:
                yk1 = constraints(yk1)
            error = yk1-yk
            error = (error*error).sum() / ((yk1*yk1).sum()+(yk*yk).sum())
            if fstart is None: fstart=error
            if error < threshold:
                break
            beta_k = (rk*rk1).sum() / (rk*rk).sum()
            pk = rk1+beta_k*pk
            yk=yk1
            rk = rk1
        if verbose:print(f"Iteration {n} | fun(t0): {fstart:.6e} | fun(terminal): {error:.6e} | step: {n:.0f} | time: {time.perf_counter()-timer:.2e}")
        return yk

    def gradient_descent(x0,
                        fun,
                        grad_fun,
                        constraints=None,
                        max_count=100,
                        check_sign=True,
                        check_der=False,
                        threshold=1e-6,
                        tol_der_error=1.,
                        verbose=True,
                        **kwargs
                        ):
        timer = time.perf_counter()
        grad = grad_fun(x0)
        grad_start = (grad*grad).sum()
        x = x0.copy()
        count = 0
        def f(t):
            next = x.copy()
            next -= grad*t
            out = fun(next)
            return out
        eps = 1e-8
        fstart,fval,xmin = None,None,0.
        while count < max_count:
            left,middle,right = 0.0,eps,2*eps
            fleft,fmiddle,fright = f(left), f(middle), f(right)
            if fstart is None: fstart,fval = fleft,fleft
            fprime = (fmiddle - fleft) / eps
            fsec = (fleft + fright - 2 * fmiddle) / (eps*eps)
            if check_der:
                consistency = np.fabs((grad*grad).sum()/fprime+1.)
                if consistency >= tol_der_error:
                    assert(False)
            if check_sign:
                #check derivative sign
                assert fprime / (np.fabs(left)+np.fabs(middle)+np.fabs(right)) <= 1e-4
            if fprime > - 1e-9: 
                break
            if fsec >0 :
                middle = -fleft / fprime
                right = -fprime / fsec
                right = np.maximum(right,2.*middle)
                fmiddle,fright=f(middle),f(right)
            else: 
                right,fright=middle,fmiddle
                middle = middle*.5
                fmiddle = f(middle*.5)
            while fleft < fmiddle-eps or fright < fmiddle-eps:
                if fleft < fmiddle-eps:
                    middle = (left + middle)*.5
                    fmiddle = f(middle)
                elif fright < fmiddle-eps:
                    left, fleft = middle, fmiddle
                    middle, fmiddle = right,fright
                    right,fright=2.*right,f(2.*right)
            if fleft <= fmiddle or fright <= fmiddle:
                break
            xmin, fval, iter, funcalls = optimize.brent(
                f, brack=(left, middle,right), maxiter=5, full_output=True
            )
            x -= grad * xmin
            if constraints is not None:
                x = constraints(x)
            count = count + 1
            grad = grad_fun(x)
            if (grad*grad).sum()/grad_start <= threshold:
                break
        if verbose:print(f"Iteration {count} | fun(t0): {fstart:.6e} | fun(terminal): {fval:.6e} | step: {xmin:.2e} | der: {fprime:.2e}, | time: {time.perf_counter()-timer:.2e}")
        return x
    
    def faiss_knn(
        X: np.ndarray, Z: np.ndarray=None, k: int = 20,metric="cosine",fun=None,**kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Nx, d = X.shape
        assert 1 <= k < Nx, "k must be in [1, N-1]"
    
        Z = X if Z is None else Z
        Nz = Z.shape[0]
        if metric == "cosine":
            X /= np.linalg.norm(X,axis=1)[:,None]  
            Z /= np.linalg.norm(Z,axis=1)[:,None]
            index = faiss.IndexFlatIP(d)
        elif metric == "euclidean":
            index = faiss.IndexFlatL2(d)
        elif metric == "METRIC_L1":
            X /= np.fabs(X).sum(1)[:,None]  
            Z /= np.fabs(Z).sum(1)[:,None]
            index = faiss.index_factory(d, f"PQ{d//28}",faiss.METRIC_L1)
            index.train(X)

        index.add(X)
        D, Id = index.search(Z, min(k, Nx))  # shapes (Nz, k+1)
        row = np.repeat(np.arange(Nz, dtype=np.int64), k)  # (N*k,)
        col = Id.reshape(-1)
        if fun is None: 
            values = D.ravel()
        else: 
            values = fun(D.ravel())
        out = sp.coo_matrix((values, (row, col)), shape=(Nz, Nx), dtype=Z.dtype).tocsr()
        return out.T # Nx, Nz


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
