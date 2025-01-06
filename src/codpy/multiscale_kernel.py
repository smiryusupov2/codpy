import sys

import numpy as np
from scipy.special import softmax

import codpy.core as core
from codpy.algs import alg
from codpy.clustering import *
from codpy.kernel import Kernel
from codpy.permutation import map_invertion


class MultiScaleKernel(Kernel):
    params = {}

    def __init__(
        self, N, n_batch=sys.maxsize, method=MiniBatchkmeans, balanced=True, **kwargs
    ):
        self.method = method
        self.N = N
        self.n_batch = n_batch
        self.balanced = balanced
        super().__init__(**kwargs)
        pass

    def set(self, x=None, fx=None, y=None, **kwargs):
        super().set(x=x, fx=fx, y=y, **kwargs)
        if self.N <= 1:
            return self
        self.clustering = self.method(
            x=self.get_x(), N=self.N, fx=self.get_fx(), **kwargs
        )
        if self.balanced:
            self.clustering = BalancedClustering(self.clustering)
        y, labels = self.clustering.cluster_centers_, self.clustering.labels_
        self.set_y(y)
        self.labels = map_invertion(labels)
        self.kernels = {}
        fx_proj = self.get_fx() - super().__call__(z=x)
        for key in self.labels.keys():
            indices = list(self.labels[key])
            if len(indices) > self.n_batch:
                N = int(len(indices) / self.n_batch) + 1
                self.kernels[key] = MultiScaleKernel(
                    x=x[indices],
                    fx=fx_proj[indices],
                    N=N,
                    n_batch=self.n_batch,
                    **kwargs,
                )
            else:
                self.kernels[key] = Kernel(x=x[indices], fx=fx_proj[indices], **kwargs)
        # test = fx - self.__call__(z=x) # reproductibility : should be zero if no regularization
        return self

    def __call__(self, z, **kwargs):
        out = super().__call__(z, **kwargs)
        if not hasattr(self, "clustering"):
            return out
        mapped_indices = self.clustering(z)
        mapped_indices = map_invertion(mapped_indices)
        for key in mapped_indices.keys():
            indices = list(mapped_indices[key])
            out[indices] += self.kernels[key](z[indices])
        return out


class MultiScaleKernelClassifier(MultiScaleKernel):
    """
    A simple overload of the kernel :class:`MultiScaleKernel` for proabability handling.
        Note:
            It overloads the prediction method as follows :

                $$\text{softmax} (\log(f)_{k,\\theta})(\cdot)$$
    """

    def set_fx(
        self,
        fx: np.ndarray,
        set_polynomial_regressor: bool = True,
        clip=alg.proportional_fitting,
        **kwargs,
    ) -> None:
        if fx is not None:
            if clip is not None:
                fx = clip(fx)
            debug = np.where(fx < 1e-9, 1e-9, fx)
            fx = np.log(debug)
        super().set_fx(fx, set_polynomial_regressor=set_polynomial_regressor, **kwargs)

    def __call__(self, z, **kwargs):
        z = core.get_matrix(z)
        if self.x is None:
            return None
            # return softmax(np.full((z.shape[0],self.actions_dim),np.log(.5)),axis=1)
        Knm = super().__call__(z, **kwargs)
        return softmax(Knm, axis=1)

    def greedy_select(
        self, N, x=None, fx=None, all=False, norm_="classifier", **kwargs
    ):
        return super().greedy_select(N=N, x=x, fx=fx, all=all, norm_=norm_, **kwargs)


class MultiScaleOT(MultiScaleKernel):
    def __init__(
        self, N, n_batch=sys.maxsize, method=MiniBatchkmeans, balanced=True, **kwargs
    ):
        super().__init__(N, n_batch, method, balanced, **kwargs)

    def set(self, x, y, epsilon=1e-8, **kwargs):
        n_clusters = self.N

        if n_clusters <= 1:
            return super().map(x=x, y=y)

        self.MB1 = self.method(x, n_clusters)
        self.MB2 = self.method(y, n_clusters)

        centers_X = self.MB1.cluster_centers_
        centers_Y = self.MB2.cluster_centers_

        C = core.KerOp.dnm(centers_X, centers_Y, distance="norm2")
        Dx = self.MB1.distance(x, centers_X) + epsilon * core.KerOp.dnm(
            x, centers_X, distance="norm22"
        )
        Dy = self.MB2.distance(y, centers_Y) + epsilon * core.KerOp.dnm(
            y, centers_Y, distance="norm22"
        )
        perm = lsap(C)
        self.label1, self.label2 = alg.two_balanced_clustering(Dx, Dy, C)
        self.invlabel1, self.invlabel2 = (
            map_invertion(self.label1),
            map_invertion(self.label2),
        )

        self.kernels = {}
        X_, Y_ = None, None
        index = 0
        for key, value in self.invlabel1.items():
            mapped_key = perm[key]
            mapped_values = self.invlabel2[mapped_key]
            X_k = x[list(value)]
            Y_k = y[list(mapped_values)]
            # a little bit violent, but safe !
            min_ = min(X_k.shape[0], Y_k.shape[0])
            X_k, Y_k = X_k[:min_], Y_k[:min_]
            k = Kernel(**kwargs)
            k.map(X_k, Y_k, **kwargs)
            if X_ is None:
                X_ = k.get_x()
                Y_ = Y_k
            else:
                X_ = np.concatenate([X_, k.get_x()])
                Y_ = np.concatenate([Y_, Y_k])
            self.kernels[key] = k
            self.label1[range(index, index + X_k.shape[0])] = key
            index += X_k.shape[0]
        Kernel.set(self, x=X_, fx=Y_, y=centers_X, **kwargs)
        return self

    def __call__(self, z, **kwargs):
        out = super().__call__(z, **kwargs)
        if not hasattr(self, "MB1"):
            return out
        labelsz = self.MB1.distance(self.get_x(), z).argmin(axis=0)
        labelsx = self.label1[labelsz]
        assign = map_invertion(labelsx)
        # extrapolation
        for key in assign.keys():
            indices = list(assign[key])
            out[indices] = self.kernels[key](z[indices])

        return out
