import numpy as np
import pandas as pd


ts_format_switchDict = { pd.DataFrame: lambda A,h,p,**kwargs :  ts_format_dataframe(A,h,p,**kwargs) }

def ts_format_np(A,h,p=0, **kwargs):
    A = np.array(A)
    dim = A.ndim
    if dim == 1:
        (first,second) = cd.tools.ts_format(get_matrix(A),h,p)
        if p ==0: return first
        return (first,second)
    if dim == 2:
        if A.shape[0] > A.shape[1]: (first,second) = cd.tools.ts_format(get_matrix(A),h,p)
        else:
            (first,second) = cd.tools.ts_format(get_matrix(A.T),h,p)
            (first,second) = (first.T,second.T)
        if p ==0: return first
        return (first,second)
    along_axis = kwargs.get('along_axis',None)
    if along_axis is not None :
        out = [ts_format_np(get_slice(A,along_axis,ind),h,p) for ind in range(0,A.shape[along_axis])]
        if p == 0 : return np.array(out)
        return np.array([s[0] for s in out]),np.array([s[1] for s in out])



def ts_format_dataframe(A,h,p, **kwargs):
    colsx, colsfx= [], []
    [ [colsx.append(col + str(n)) for col in A.columns] for n in range(0,h) ]
    [ [colsfx.append(col + str(n)) for col in A.columns] for n in range(0,p) ]

    ts_format_param = kwargs.get('ts_format',None)

    if ts_format_param is not None:
        x_csv = ts_format_param.get('x_csv',None)
        if x_csv is not None: x.to_csv(x_csv, sep = ';', index=False)
        fx_csv = ts_format_param.get('fx_csv',None)
        if fx_csv is not None: fx.to_csv(fx_csv, sep = ';', index=False)

    x,fx = ts_format_np(get_matrix(A),h,p,**kwargs)
    x,fx = pd.DataFrame(x,columns=colsx, index = A.index[:x.shape[0]]),pd.DataFrame(fx,columns=colsfx,index = A.index[h:fx.shape[0]+h])
    return x, fx

def ts_format_diff(A,h,p = 0, **kwargs):
    A.values
    diff = A.diff(axis=0)
    diff.values
    diff = diff.dropna(axis = 'rows')
    x,fx = ts_format(A,h,p, **kwargs)
    diffx,difffx = ts_format(diff,h,p, **kwargs)
    return x.iloc[:-1,:], difffx

def ts_format(x,h,p, **kwargs):
    type_debug = type(x)
    method = ts_format_switchDict.get(type_debug,lambda x,h,p,**kwargs: ts_format_np(x,h,p,**kwargs))
    if method is not None: return method(x,h,p,**kwargs)