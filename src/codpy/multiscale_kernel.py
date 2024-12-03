import sys

import numpy as np
from scipy.special import softmax

import codpy.core as core
from codpy.clustering import BalancedClustering, MiniBatchkmeans
from codpy.kernel import Kernel, clip_probs
from codpy.permutation import lsap, map_invertion


class MultiScaleKernel(Kernel):
    params = {}

    def __init__(
        self, N, n_batch=sys.maxsize, method=MiniBatchkmeans, balanced=True, **kwargs
    ):
        self.method = method
        self.N = N
        self.n_batch = n_batch
        self.balanced = balanced
        super().__init__(method=self.method, **kwargs)
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
        self, fx: np.ndarray, set_polynomial_regressor: bool = True, **kwargs
    ) -> None:
        if fx is not None:
            fx = np.log(clip_probs(fx))
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


class MultiScaleT(MultiScaleKernel):
    def __init__(
        self, N, n_batch=sys.maxsize, method=MiniBatchkmeans, balanced=True, **kwargs
    ):
        super().__init__(N, n_batch, method, balanced, **kwargs)

    def set(self, X1, X2):
        n_clusters = self.N

        if n_clusters <= 1:
            return super().set(x=X1, fx=X2)

        self.MB1 = self.method(X1, n_clusters)
        self.MB2 = self.method(X2, n_clusters)
        if self.balanced:
            self.clustering1 = BalancedClustering(self.MB1)
            self.clustering2 = BalancedClustering(self.MB2)

        centers_X1 = self.clustering1.cluster_centers_
        centers_X2 = self.clustering2.cluster_centers_

        C = self.clustering1.distance(centers_X1, centers_X2)
        perm = lsap(C)
        label1, label2 = self.clustering1.get_labels(), self.clustering2.get_labels()
        invlabel1, invlabel2 = map_invertion(label1), map_invertion(label2)

        self.kernels = {}
        for key, value in invlabel1.items():
            mapped_key = perm[key]
            mapped_values = invlabel2[mapped_key]
            X_k = X1[list(value)]
            Y_k = X2[list(mapped_values)]

            if X_k.shape[0] > Y_k.shape[0]:
                idx = np.random.choice(X_k.shape[0], size=Y_k.shape[0], replace=False)
                X_k = X_k[idx]
                Dist = self.clustering1.distance(Y_k, X_k)
                Y_k = Y_k[lsap(Dist)]
            elif Y_k.shape[0] > X_k.shape[0]:
                idx = np.random.choice(Y_k.shape[0], size=X_k.shape[0], replace=False)
                Y_k = Y_k[idx]
                Dist = self.clustering1.distance(X_k, Y_k)
                X_k = X_k[lsap(Dist)]

            k = Kernel()
            k.map(X_k, Y_k)
            self.kernels[key] = k

    def __call__(self, z, **kwargs):
        if not hasattr(self, "clustering1"):
            return super().__call__(z, **kwargs)
        labels1 = self.clustering1(z)
        assign = map_invertion(labels1)
        # extrapolation
        z_mapped = np.zeros_like(z)
        for key in assign.keys():
            indices = list(assign[key])
            z_mapped[indices] = self.kernels[key](z[indices])

        return z_mapped
