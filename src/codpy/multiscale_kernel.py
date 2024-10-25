import codpy.core as core
from codpy.kernel import *
from codpy.algs import alg
import numpy as np
import pandas as pd
from scipy.special import softmax
from codpy.permutation import map_invertion
from codpy.clustering import GreedySearch,MiniBatchkmeans


class MultiScaleKernel(Kernel):
    params = {}
    def __init__(self,N,method = MiniBatchkmeans,**kwargs):
        self.method = method
        self.N = N
        super().__init__(**kwargs)


    def set(self,x=None,fx=None,y=None,**kwargs):
        super().set(x=x,fx=fx,y=y,**kwargs)
        self.clustering = self.method(x=self.get_x(),N=self.N,fx=self.get_fx(),**kwargs)
        y,labels = self.clustering.cluster_centers_,self.clustering.labels_
        self.labels = map_invertion(labels)
        self.kernels = {}
        fx_proj = self.get_fx() - super().__call__(z=x) 
        for key in self.labels.keys():
            indices = list(self.labels[key])
            self.kernels[key] = Kernel(x=x[indices],fx=fx_proj[indices])
        # test = fx - self.__call__(z=x) # reproductibility : should be zero if no regularization
        return self

    def __call__(self, z, **kwargs):
        out = super().__call__(z,**kwargs)
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
    def set_fx(self, fx: np.ndarray, set_polynomial_regressor: bool = True, **kwargs  ) -> None:
        if fx is not None: fx = np.log(clip_probs(fx))
        super().set_fx(fx,set_polynomial_regressor=set_polynomial_regressor,**kwargs)
    def __call__(self, z, **kwargs):
        z = core.get_matrix(z)
        if self.x is None : return None
            # return softmax(np.full((z.shape[0],self.actions_dim),np.log(.5)),axis=1)
        Knm= super().__call__(z,**kwargs)
        return softmax(Knm,axis=1)
    
    def greedy_select(self, N,x=None, fx=None, all=False, norm_="classifier", **kwargs):
        return super().greedy_select(N=N,x=x,fx=fx, all=all, norm_=norm_, **kwargs)
