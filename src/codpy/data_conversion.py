import xarray
import pandas as pd
import numpy as np
import datetime


get_data_switchDict = { pd.DataFrame: lambda vals :  vals.values,
                        pd.Series: lambda vals : np.array(vals.array, dtype= 'float'),
                        # torch.Tensor: lambda vals : vals.detach().numpy(),
                        tuple: lambda xs : [get_data(vals) for vals in xs],
                        xarray.core.dataarray.DataArray: lambda xs : get_data(xs.values) 
                    }

get_date_switchDict = { str: lambda vals,**kwargs :  datetime.datetime.strptime(vals, kwargs.get('date_format','%d/%m/%Y')).date(),
                        datetime.date:lambda vals,**kwargs : vals,
                        int : lambda vals,**kwargs : get_date(datetime.date.fromordinal(vals)),
                        float : lambda vals,**kwargs : get_date(int(vals)),
                        np.float64 : lambda vals,**kwargs : get_date(int(vals)),
                        pd._libs.tslibs.timestamps.Timestamp : lambda vals,**kwargs : pd.to_datetime(vals),
                        # torch.Tensor : lambda vals,**kwargs: get_date(get_float(vals)),
                        np.ndarray : lambda vals,**kwargs: get_date(get_float(vals)),
                        datetime.datetime: lambda vals,**kwargs: vals,
                    }


get_float_switchDict = {
                        #QuantLib.QuantLib.Date: lambda vals, **kwargs: get_float(datetime.datetime(vals.year(), vals.month(), vals.dayOfMonth())),
                        pd.DataFrame: lambda vals, **kwargs: get_data(vals),
                        pd._libs.tslibs.timedeltas.Timedelta: lambda vals, **kwargs: float(vals/datetime.timedelta(days=1)),
                        pd._libs.tslibs.timestamps.Timestamp : lambda vals, **kwargs: get_float(vals.to_pydatetime(), **kwargs),
                        pd._libs.tslibs.nattype.NaTType : lambda vals, **kwargs: np.nan,
                        pd.core.indexes.base.Index : lambda vals, **kwargs: get_float(vals.tolist(), **kwargs),
                        pd.core.series.Series : lambda vals, **kwargs: get_float(vals.tolist(), **kwargs),
                        pd.core.indexes.datetimes.DatetimeIndex: lambda vals, **kwargs: get_float(vals.tolist(), **kwargs),
                        pd.core.indexes.numeric.Float64Index: lambda vals, **kwargs: get_float(vals.tolist(), **kwargs),
                        pd.Index:lambda vals, **kwargs: get_float(vals.tolist(), **kwargs),
                        datetime.date :lambda vals, **kwargs: float(vals.toordinal()),
                        datetime.datetime: lambda vals, **kwargs: vals.timestamp(),#/86400, #number of seconds in a day, to keep data and datetime consistent
                        datetime.timedelta: lambda vals, **kwargs: vals/datetime.timedelta(days=1),
                        list: lambda vals, **kwargs:  [get_float(z, **kwargs) for z in vals],
                        type({}.keys()) : lambda vals, **kwargs: get_float(list(vals), **kwargs),
                        type({}.values()) : lambda vals, **kwargs: get_float(list(vals), **kwargs),
                        str: lambda vals, **kwargs: get_float(get_date(vals.replace(',','.'),**kwargs),**kwargs),
                        np.array : lambda vals, **kwargs: np.array([get_float(z, **kwargs) for z in vals]),
                        np.ndarray : lambda vals, **kwargs: np.array([get_float(z, **kwargs) for z in vals]),
                        xarray.core.dataarray.DataArray: lambda xs, **kwargs : get_float(xs.values),
                        }

my_len_switchDict = {list: lambda vals: len(vals),
                    pd.core.indexes.base.Index  : lambda vals: len(vals),
                    np.array : lambda vals: len(vals),
                    np.ndarray : lambda vals: vals.size,
                    pd.DataFrame  : lambda vals: my_len(vals.values),
                    pd.core.groupby.generic.DataFrameGroupBy : lambda vals: vals.ngroups
                    }

def get_data(x,dtype='float'):
    type_debug = type(x)
    method = get_data_switchDict.get(type_debug,lambda x: np.asarray(x,dtype=dtype))
    return method(x)

def get_float_nan(vals, **k):
    if isinstance(vals,list):return [get_float_nan(y,**k) for y in vals]
    if isinstance(vals,np.ndarray):return [get_float_nan(y,**k) for y in vals]
    out = float(vals)
    # out = np.array(x, dtype=float)
    nan_val = k.get("nan_val",np.nan)
    if pd.isnull(out) : 
        out = nan_val
    return out
def Id(vals,**k):return vals
def get_float(vals, **kwargs):
    type_ = type(vals)
    method = get_float_switchDict.get(type_,Id)
    temp = method(vals,**kwargs)
    out = get_float_nan(temp,**kwargs)
    return out

def get_date(vals,**kwargs):
    if isinstance(vals,list): return [get_date(n,**kwargs) for n in x]
    type_debug = type(vals)
    method = get_date_switchDict.get(type_debug,None)
    return method(vals,**kwargs)

def my_len(x):
    if is_primitive(x):return 0
    type_ = type(x)
    method = my_len_switchDict.get(type_,lambda x : 0)
    return method(x)

primitive = (int, str, bool,np.float32,np.float64,float)
def is_primitive(thing):
    debug = type(thing)
    return isinstance(thing, primitive)

def get_matrix(x,dtype='float'):
    if x is None: return []
    if isinstance(x,list): 
        if len(x)==0: return []
        test = [get_matrix(y).T for y in x]
        return np.concatenate(test)
    # if isinstance(x,list): return np.array([np.array(y) for y in x])
    if isinstance(x,tuple): return [get_matrix(y) for y in x]
    x = get_data(x,dtype=dtype)
    if is_primitive(x) : return x
    dim = len(x.shape)
    if dim==2:return x
    if dim==1:return np.reshape(x,[len(x),1])
    if dim==0:return np.reshape(x,[1,1])

    raise AssertionError("get_matrix accepts only vector or matrix entries. Dim(x)="+str(dim))

def get_slice(my_array, dimension, index):
    items = [slice(None, None, None)] * my_array.ndim
    def helper(d,ind):
        items[d] = ind
    if isinstance(dimension,list): [helper(d,ind) for d,ind in zip(dimension,index)]
    else:helper(dimension,index)
    array_slice = my_array[tuple(items)]
    return array_slice

def tensor_vectorize(fun, x):
    N, _ = x.shape[0],x.shape[1]
    E = len(fun(x[0]))
    out = np.zeros((N, E))
    for n in range(0, N):
        out[n] = fun(x[n])
    return out.reshape(N,E,1)

def matrix_vectorize(fun, x):
    N, _ = x.shape[0],x.shape[1]
    out = np.zeros((N, 1))
    def helper(n) : out[n,0] = fun(x[n])
    [helper(n) for n in range(0, N)]
    return out

def array_helper(x):
    if np.ndim(x) == 1:
        return np.reshape(len(x),1)
    return x