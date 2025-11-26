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
import scipy.fft
import scipy.linalg
from scipy.special import softmax
from scipy import optimize
from scipy import signal
from scipy.sparse import lil_array,csr_matrix


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

    def adams_pytorch(x0,
                        fun,
                        **kwargs
                        ):
        raise NotImplementedError("adams not implemented yet")

    def gradient_descent(x0,
                        fun,
                        grad_fun,
                        constraints=None,
                        max_count=10,
                        check_sign=True,
                        check_der=False,
                        threshold=1e-6,
                        tol_der_error=1.,
                        verbose=True,
                        eps = 1e-8,
                        **kwargs
                        ):
        timer = time.perf_counter()
        grad = grad_fun(x0,**kwargs)
        grad_start = (grad*grad).sum()
        if grad_start <= threshold:
            if verbose:print(f"gradient_descent : No grad")
            return x0
        x = x0.copy()
        count = 0
        def f(t):
            next = x.copy()
            next -= grad*t
            out = fun(next)
            return out
        fstart,fval,xmin = None,None,0.
        while count < max_count:
            left,middle,right = 0.0,eps,2*eps
            fleft,fmiddle,fright = f(left), f(middle), f(right)
            while fmiddle >= fleft:
                if fleft > fmiddle:
                    left,fleft = eps,fmiddle
                eps *=2.
                if eps >= 1.:
                    break
                middle,right = eps,2*eps
                fmiddle,fright = fright, f(right)
            if fstart is not None and fleft > fval:
                pass
                # break
            if fstart is None: fstart,fval = fleft,fleft
            fprime = (fmiddle - fleft) / (middle-left)
            fsec = (fleft + fright - 2 * fmiddle) / (eps*eps)
            consistency = np.fabs((grad*grad).sum()/fprime+1.)
            if check_der:
                if consistency >= tol_der_error:
                    assert(False)
            if check_sign:
                #check derivative sign
                if fprime / (np.fabs(fleft)) >= 1e-4:
                    break
                    assert(False)
            if fprime > - 1e-9: 
                break
            if fsec >0 and consistency < 1.:
                middle = -fleft / fprime
                right = -fprime / fsec
                right = np.maximum(right,2.*middle)
                fmiddle,fright=f(middle),f(right)
            # else: 
            #     right,fright=middle,fmiddle
            #     middle = middle*.5
            #     fmiddle = f(middle)
            while fleft < fmiddle-1e-9 or fright < fmiddle-1e-9:
                if fleft < fmiddle-1e-9:
                    middle = (left + middle)*.5
                    fmiddle = f(middle)
                elif fright < fmiddle-1e-9:
                    left, fleft = middle, fmiddle
                    middle, fmiddle = right,fright
                    right,fright=2.*right,f(2.*right)
            if fleft <= fmiddle or fright <= fmiddle:
                if fleft <= fmiddle and fleft <= fmiddle:
                    break
                xmin = right
            else:
                xmin, fval, iter, funcalls = optimize.brent(
                    f, brack=(left, middle,right), maxiter=5, full_output=True
                )
            x -= grad * xmin
            if constraints is not None:
                x = constraints(x)
            count = count + 1
            grad = grad_fun(x,**kwargs)
            if (grad*grad).sum()/grad_start <= threshold:
                break
        if verbose:print(f"gradient_descent : Iteration {count} | fun(t0): {fstart:.6e} | eps : {eps:.6e} fun(terminal): {fval:.6e} | step: {xmin:.2e} | time: {time.perf_counter()-timer:.2e}  | der: {fprime:.2e}, consistency: {consistency:.2e}")
        return x
    
    def sparse_softmax(mat, axis=1):
        out = mat.copy()
        def helper(n):
            indices = mat.indices[mat.indptr[n]:mat.indptr[n+1]]
            out.data[indices] = softmax(mat.data[indices])
        map(helper, list(range(mat.indptr.shape[0]-1)))
        return out
    
    def faiss_make_index_max(x, z=None, metric="cosine", **kwargs):
        """
            Faiss index creation. 
            Args:
                x (np.ndarray): Data points of shape (Nx, d).
                z (np.ndarray, optional): Query points of shape (Nz, d). If None, uses x.
            Returns:
                faiss.Index: Faiss index object.

        """
        Nx, d = x.shape
        Z = x if z is None else z
        Nz = Z.shape[0]
        if metric == "cosine":
            X = x/(np.linalg.norm(x,axis=1)[:,None]+1e-9)  
            Z = Z/(np.linalg.norm(Z,axis=1)[:,None]+1e-9)
            index = faiss.IndexFlatIP(d)
        elif metric == "euclidean":
            X = x 
            index = faiss.IndexFlatL2(d)
        elif metric == "METRIC_L1":
            X = x/(np.fabs(x).sum(1)[:,None]+1e-9 ) 
            Z = Z/(np.fabs(Z).sum(1)[:,None]+1e-9)
            index = faiss.index_factory(d, f"PQ{d//28}",faiss.METRIC_L1)
            index.train(X)

        index.add(X)
        return index 
    
    def faiss_knn_search_max(x, index, z=None, k = 20, faiss_fun=None, **kwargs):
        """
            Faiss k-nearest neighbors search using a pre-built index.
            Args:
                x: Data points of shape (Nx, d).
                index: Pre-built Faiss index.
                z: Query points of shape (Nz, d). If None, uses x.
                k: Number of nearest neighbors to find.
        """
        Nx, d = x.shape
        k = min(k, Nx - 1)
        Z = x if z is None else z
        Nz = Z.shape[0]

        D, Id = index.search(Z, min(k, Nx))  # shapes (Nz, k+1)
        row = np.repeat(np.arange(Nz, dtype=np.int64), k)  # (N*k,)
        col = Id.reshape(-1)
        if faiss_fun is None: 
            values = D.ravel()
        else: 
            values = faiss_fun(D.ravel())
        out = sp.coo_matrix((values, (row, col)), shape=(Nz, Nx), dtype=Z.dtype).tocsr()
        return out.T # Nx, Nz

    def faiss_knn_index(
        x: np.ndarray, k: int = 20,metric="cosine",faiss_fun=None,**kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Nx, d = x.shape
        if metric == "cosine":
            x = x/(np.linalg.norm(x,axis=1)[:,None]+1e-9)
            index = faiss.IndexFlatIP(d)
        elif metric == "euclidean":
            X = x 
            index = faiss.IndexFlatL2(d)
        elif metric == "METRIC_L1":
            x = x/(np.fabs(Z).sum(1)[:,None]+1e-9)
            index = faiss.index_factory(d, f"PQ{d//28}",faiss.METRIC_L1)
            index.train(x)

        index.add(x)
        return index # Nx, Nz
    def faiss_knn_search(
        z: np.ndarray, k: int = 20,metric="cosine",index=None,**kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if metric == "cosine":
            z = z/(np.linalg.norm(z,axis=1)[:,None]+1e-9)
        elif metric == "METRIC_L1":
            z = z/(np.fabs(Z).sum(1)[:,None]+1e-9)
        if index is None:
            index = Alg.faiss_knn_index(x=z,k=k,metric=metric,**kwargs)
        Nx = index.ntotal
        k = min(k, Nx - 1)

        D, Id = index.search(z, k)
        return D,Id,index  # shapes (Nz, k+1)
    
    def faiss_knn(
        z: np.ndarray, k: int = 20,metric="cosine",faiss_fun=None,index=None,**kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Nx = index.ntotal
        Nz = z.shape[0]
        k = min(k, Nx - 1)

        D, Id, index = Alg.faiss_knn_search(z=z,k=k,metric=metric,index=index,**kwargs)
        col = np.repeat(np.arange(Nz, dtype=np.int64), k)  # (N*k,)
        row = Id.reshape(-1)
        if faiss_fun is None: 
            values = D.ravel()
        else: 
            values = faiss_fun(D.ravel())
        out = sp.coo_matrix((values, (col,row)), shape=(Nz, Nx), dtype=z.dtype).tocsr()
        return out,index # Nx, Nz

    def faiss_knn_select(x: np.ndarray, faiss_batch_size=1,metric = "cosine",faiss_threshold=1e-1,faiss_nb_select=None,faiss_fun=None,**kwargs):
        mask = np.where((x*x).sum(1) > 1e-9)[0]
        x = x[mask]
        Nx, d = x.shape
        timer = time.perf_counter()
        if metric == "cosine":
            x = x/(np.linalg.norm(x,axis=1)[:,None]+1e-9)
            index = faiss.IndexFlatIP(d)
        elif metric == "euclidean":
            index = faiss.IndexFlatL2(d)
        elif metric == "METRIC_L1":
            x = x/(np.fabs(x).sum(1)[:,None]+1e-9)
            index = faiss.index_factory(d, f"PQ{d//28}",faiss.METRIC_L1)
            index.train(x)
        def helper(n):
            z = x[n:min(n+faiss_batch_size,Nx)]
            if metric == "cosine":
                mask = np.where((z*z).sum(1) > 1e-9)[0]
                z = z[mask]
                if z.shape[0] == 0:
                    return None
                if index.ntotal == 0:
                    index.add(z)
                    return z
                D, Id = index.search(z, 1)
                if faiss_fun is not None:
                    D = faiss_fun(D)
                mask = np.where(D[:,0] > faiss_threshold)[0]
                if mask.size > 0:
                    index.add(z[mask])
                    return z[mask]
        out = map(helper, list(range(0, Nx,faiss_batch_size)))
        out = np.concatenate([o for o in out if o is not None])
        if faiss_nb_select is not None and out.shape[0] > faiss_nb_select:
            D, Id = index.search(out, 2)
            if faiss_fun is not None:
                D = faiss_fun(D)
            indices = np.argsort(-D[:,1])
            out = out[indices[:faiss_nb_select]]    
            index.remove_ids(indices[faiss_nb_select:])
            print(f"faiss_knn_select: time {time.perf_counter()-timer} seconds.")
        return out,index # Nx, Nz

    def grad_faiss_knn(x: np.ndarray, z: np.ndarray=None, k: int = 20, knm=None,metric="cosine",grad_faiss_fun=None,grad_threshold=1e-9,**kwargs):
        D = x.shape[1]
        if knm is None:
            knm = Alg.faiss_knn(x,z,k=k,**kwargs)
        if metric == "cosine":
            x = x/(np.linalg.norm(x,axis=1)[:,None]+1e-9)  
            z = z/(np.linalg.norm(z,axis=1)[:,None]+1e-9)
            indptr,indices,data = cd.alg.grad_faiss(knm.indptr,knm.indices,knm.data,x,z,grad_threshold)
            out = csr_matrix((data,indices,indptr), shape=(x.shape[0]*D,z.shape[0]),dtype=knm.dtype)
            pass
        elif metric == "euclidean":
            assert(False,"Not implemented yet")
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
