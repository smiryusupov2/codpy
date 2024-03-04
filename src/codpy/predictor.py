import pandas as pd
import numpy as np
from functools import partial, cache
import itertools
from core import *
from algs import *
import warnings

class greedy_nys_predictor:
    params = {}
    def __init__(self,**kwargs):
        self.params = kwargs.copy()
        self.x = pd.DataFrame(kwargs['x'])
        self.fx = pd.DataFrame(kwargs['fx'])
        self.z = pd.DataFrame(kwargs.get("z",None))
        self.params['x'],self.params['y'],self.params['fy'] = self.x,[],[]
        self.knm_inv, self.indices =  alg.HybridGreedyNystroem(x=self.x,fx=self.fx,**self.params)
        self.x,self.fx = self.x.iloc[self.indices],self.fx.iloc[self.indices]
        self.params['x'] = self.x
        self.params['y'] = self.x
        self.params['fx'] = self.fx
        self.mappedx = kernel.map(self.x)
        if self.z is not None: self.mappedz = kernel.map(self.z)
        self.order = cd.kernel.get_polynomial_order()
        self.reg = cd.kernel.get_regularization()
        self.kernel =  kernel.get_kernel_ptr()
        self.map =  kernel.get_map_ptr()
    def __call__(self, **kwargs):
        y,z = get_matrix(self.params['y']),get_matrix(kwargs['z'])
        # return op.projection(**{**self.params,**{'x':y,'y':y,'z':z,'fx':kwargs['fx']}}) #for debug
        kernel.set_kernel_ptr(self.kernel)
        cd.kernel.set_map_ptr(self.map)
        cd.kernel.set_polynomial_order(self.order)
        cd.kernel.set_regularization(self.reg)
        Knm= op.Knm(**{**self.params,**{'x':z,'y':y,'fy':self.knm_inv,'set_codpykernel':None,'rescale':False}})
        return Knm


if __name__ == "__main__":

    greedy_nys_predictor(x=np.random.rand(10,5),fx=np.random.rand(10,7))
    pass

