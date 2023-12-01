import numpy as np
import numdifftools.nd_scipy as nd
import pandas as pd
from codpy.utils.data_conversion import get_matrix
import copy

class FD:
    def nabla(x,fun,set_fun=None,**kwargs): 
        if isinstance(x,list): return [FD.nabla(x=x[n],fun=fun,**kwargs) for n in range(0,x.shape[0])]
        if set_fun is None: set_fun = lambda x,**k : k
        if x.ndim == 2 : 
            return np.concatenate([FD.nabla(x=x[n],fun=fun,set_fun=set_fun,**kwargs) for n in range(0,x.shape[0])], axis = 0)
        def lambda_helper(x,**kwargs):
            k = set_fun(x=x,**kwargs)
            out = fun(**k)
            return out
        out = nd.Gradient(lambda_helper)(x,**kwargs)
        # print("grad:",get_matrix(out).T)
        return get_matrix(out).T

    def hessian(x,fun,**kwargs): 
        # if fun == None: fun = option_param.price
        if isinstance(x,list): return [FD.hessian(x=x[n],fun=fun,**kwargs) for n in range(0,value.shape[0])]
        if x.ndim == 2 : 
            out = np.array([FD.hessian(x=x[n],fun=fun,**kwargs) for n in range(0,x.shape[0])])
            return out
        def hessian_helper(x,**kwargs):
            out = FD.nabla(x=x,fun=fun,**kwargs)
            out = np.squeeze(out)
            return out

        out = nd.Jacobian(hessian_helper)(x=x,**kwargs)
        # print("grad:",get_matrix(out).T)
        return out.T

    def nablas(fun, x,**kwargs): 
        copy_kwargs = copy.deepcopy(kwargs)
        # copy_kwargs = kwargs.copy()
        def helper(v): return FD.nabla(fun = fun, x = v,**copy_kwargs) 
        if isinstance(x,list): out = [helper(v) for v in get_matrix(x)]
        elif isinstance(x,np.ndarray): 
            if x.ndim == 1: return helper(x)
            out = [helper(x[n]) for n in range(0,x.shape[0]) ]
        elif isinstance(x,pd.DataFrame): return FD.nablas(fun=fun,x= x.values,**copy_kwargs)
        out = np.array(out).reshape(values.shape)
        return out

    def hessians(fun, x,**kwargs): 
        copy_kwargs = copy.deepcopy(kwargs)
        # copy_kwargs = kwargs.copy()
        def helper(v): return FD.hessian(fun = fun, x = v,**copy_kwargs) 
        if isinstance(x,list): out = [helper(v) for v in get_matrix(x)]
        elif isinstance(x,np.ndarray): 
            if x.ndim == 1: return helper(x)
            out = [helper(x[n]) for n in range(0,x.shape[0]) ]
        elif isinstance(x,pd.DataFrame): return FD.hessians(fun=fun,x= x.values,**copy_kwargs)
        out = np.array(out).reshape(x.shape[0],x.shape[1],x.shape[1])
        return out