import codpy.core as core
from codpy.kernel import *
from codpy.algs import alg
import numpy as np
from scipy.special import softmax
from codpy.permutation import map_invertion
from sklearn.cluster import KMeans,MiniBatchKMeans

class MiniBatchkmeans(MiniBatchKMeans):
    def __init__(self,x, N, max_iter = 300, random_state = 42,batch_size = 1024,verbose = False,**kwargs):
        super().__init__(n_clusters=N,
            init="k-means++",
            batch_size = batch_size,
            verbose = verbose,
            max_iter=max_iter,
            random_state=random_state)
        self.fit(x)
    def __call__(self,z, **kwargs):
        return self.predict(z)

class GreedySearch(Kernel):
    def __init__(self,x, N, **kwargs):
        # super().__init__(x=x,max_nystrom=N,**kwargs) #seemed to be a better idea, but no evidence in results !
        super().__init__(x=x,**kwargs)
        self.greedy_select(N=N,x=x,**kwargs,all=True)
        self.cluster_centers_ = self.get_x()[self.indices]
        self.labels_ = self(self.get_x())
    def __call__(self,z, **kwargs):
        self.set_kernel_ptr()
        labels = core.op.Dnm(z, self.cluster_centers_).argmin(axis=1)
        return labels
    
class SharpDiscrepancy(GreedySearch):    
    def __init__(self,x, N, itermax = 10, **kwargs):
        super().__init__(x=x,N=N,**kwargs)
        self.cluster_centers_ = cd.alg.sharp_discrepancy(self.get_x(),N,itermax)
        self.labels_ = self(self.get_x())
