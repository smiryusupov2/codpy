import codpy.core as core
from codpy.kernel import *
from codpy.algs import alg
import numpy as np
from scipy.special import softmax
from codpy.permutation import map_invertion
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
                                            
class MiniBatchkmeans(MiniBatchKMeans):
    def __init__(self,x, N, max_iter = 300, random_state = 42,batch_size = 1024,verbose = False,**kwargs):
        super().__init__(n_clusters=N,
            init="k-means++",
            batch_size = batch_size,
            verbose = verbose,
            max_iter=max_iter,
            random_state=random_state)
        self.fit(x)
        self.x = x
        self.indices = self.distance(self.cluster_centers_,self.x).argmin(axis=1)
    def get_labels(self):
        return self(self.x)
    def __call__(self,z, **kwargs):
        return self.predict(z)
    def distance(self,x,y):
        return core.op.Dnm(x, y, distance="norm22")

class GreedySearch(Kernel):
    def __init__(self,x, N, **kwargs):
        # super().__init__(x=x,max_nystrom=N,**kwargs) #seemed to be a better idea, but no evidence in results !
        super().__init__(x=x,**kwargs)
        self.greedy_select(N=N,x=x,**kwargs)
        self.cluster_centers_ = x[self.indices]
    def get_labels(self):
        return self(self.x)
    def __call__(self,z, **kwargs):
        self.set_kernel_ptr()
        labels = core.op.Dnm(z, self.cluster_centers_).argmin(axis=1)
        return labels
    def distance(self,x,y):
        self.set_kernel_ptr()
        return core.op.Dnm(x, y)
    
class SharpDiscrepancy(GreedySearch):    
    def __init__(self,x, N, itermax = 10, **kwargs):
        super().__init__(x=x,N=N,**{**kwargs,**{"all":True}})
        self.cluster_centers_ = cd.alg.sharp_discrepancy(self.get_x(),N,itermax)

class BalancedClustering:
    def __init__(self,method,epsilon=0,**kwargs):
        self.cut = 100000000
        self.method = method
        self.cluster_centers_ = self.method.cluster_centers_
        self.x = self.method.x
        self.D = self.method.distance(self.x,self.cluster_centers_) + core.op.Dnm(self.x,self.cluster_centers_, distance="norm22")*epsilon
        self.labels_ = np.array(cd.alg.balanced_clustering(self.D))
        pass
    def get_labels(self):
        return self.labels_

    def __call__(self,z, **kwargs):
        nb = (int) (z.shape[0]*self.x.shape[0] / self.cut)+1
        cut = (int) (z.shape[0] / nb) + 1
        labels = None
        for n in range(nb):
            locals = list(self.method.distance(z[n*cut:min((n+1)*cut,z.shape[0])], self.x).argmin(axis=1))
            locals = self.labels_[locals]
            if labels is None: 
                labels=locals.copy()
            else:
                labels = np.concatenate([labels,locals])
        return labels
