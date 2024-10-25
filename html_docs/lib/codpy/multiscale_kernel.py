import codpy.core as core
from codpy.kernel import *
from codpy.algs import alg
import numpy as np
import pandas as pd
from scipy.special import softmax
from codpy.permutation import map_invertion
from sklearn.cluster import KMeans,MiniBatchKMeans

 
class MiniBatchkmeans(MiniBatchKMeans):
    def __init__(self,x, N, max_iter = 300, random_state = 42,batch_size = 8192,verbose = False,**kwargs):
        super().__init__(n_clusters=N,
            init="k-means++",
            batch_size = batch_size,
            verbose = verbose,
            max_iter=max_iter,
            random_state=random_state)
        self.fit(x)
    def __call__(self,z, **kwargs):
        return self.predict(z)



class MultiScaleKernel(Kernel):
    params = {}
    def __init__(self,N,method = MiniBatchkmeans,**kwargs):
        self.method = method
        self.N = N
        super().__init__(**kwargs)


    def set(self,x=None,fx=None,y=None,**kwargs):
        self.clustering = self.method(x=x,N=self.N,fx=fx,**kwargs)
        y,labels = self.clustering.cluster_centers_,self.clustering.labels_
        self.labels = map_invertion(labels)
        super().set(x=x,fx=fx,y=y,**kwargs)
        self.kernels = {}
        fx_proj = fx- super().__call__(z=x) 
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
