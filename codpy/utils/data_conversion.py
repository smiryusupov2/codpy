import xarray
import pandas as pd
import numpy as np
import datetime


get_data_switchDict = { pd.DataFrame: lambda x :  x.values,
                        pd.Series: lambda x : np.array(x.array, dtype= 'float'),
                        # torch.Tensor: lambda x : x.detach().numpy(),
                        tuple: lambda xs : [get_data(x) for x in xs],
                        xarray.core.dataarray.DataArray: lambda xs : get_data(xs.values) 
                    }

get_date_switchDict = { str: lambda x,**kwargs :  datetime.datetime.strptime(x, kwargs.get('date_format','%d/%m/%Y')),
                        datetime.date:lambda x,**kwargs : x,
                        int : lambda x,**kwargs : get_date(datetime.date.fromordinal(x)),
                        float : lambda x,**kwargs : get_date(int(x)),
                        np.float64 : lambda x,**kwargs : get_date(int(x)),
                        pd._libs.tslibs.timestamps.Timestamp : lambda x,**kwargs : pd.to_datetime(x),
                        # torch.Tensor : lambda x,**kwargs: get_date(get_float(x)),
                        np.ndarray : lambda x,**kwargs: get_date(get_float(x)),
                        datetime.datetime: lambda x,**kwargs: x,
                    }


get_float_switchDict = {
                        #QuantLib.QuantLib.Date: lambda x, **kwargs: get_float(datetime.datetime(x.year(), x.month(), x.dayOfMonth())),
                        pd.DataFrame: lambda x, **kwargs: get_data(x),
                        pd._libs.tslibs.timedeltas.Timedelta: lambda x, **kwargs: float(x/datetime.timedelta(days=1)),
                        pd._libs.tslibs.timestamps.Timestamp : lambda x, **kwargs: get_float(x.to_pydatetime(), **kwargs),
                        pd._libs.tslibs.nattype.NaTType : lambda x, **kwargs: np.nan,
                        pd.core.indexes.base.Index : lambda x, **kwargs: get_float(x.tolist(), **kwargs),
                        pd.core.series.Series : lambda x, **kwargs: get_float(x.tolist(), **kwargs),
                        pd.core.indexes.datetimes.DatetimeIndex: lambda x, **kwargs: get_float(x.tolist(), **kwargs),
                        pd.core.indexes.numeric.Float64Index: lambda x, **kwargs: get_float(x.tolist(), **kwargs),
                        pd.Index:lambda x, **kwargs: get_float(x.tolist(), **kwargs),
                        datetime.date :lambda x, **kwargs: float(x.toordinal()),
                        datetime.datetime: lambda x, **kwargs: x.timestamp(),#/86400, #number of seconds in a day, to keep data and datetime consistent
                        datetime.timedelta: lambda x, **kwargs: x/datetime.timedelta(days=1),
                        list: lambda x, **kwargs:  [get_float(z, **kwargs) for z in x],
                        type({}.keys()) : lambda x, **kwargs: get_float(list(x), **kwargs),
                        type({}.values()) : lambda x, **kwargs: get_float(list(x), **kwargs),
                        str: lambda x, **kwargs: get_float(get_date(x.replace(',','.'),**kwargs),**kwargs),
                        np.array : lambda x, **kwargs: np.array([get_float(z, **kwargs) for z in x]),
                        np.ndarray : lambda x, **kwargs: np.array([get_float(z, **kwargs) for z in x]),
                        xarray.core.dataarray.DataArray: lambda xs, **kwargs : get_float(xs.values),
                        }

my_len_switchDict = {list: lambda x: len(x),
                    pd.core.indexes.base.Index  : lambda x: len(x),
                    np.array : lambda x: len(x),
                    np.ndarray : lambda x: x.size,
                    pd.DataFrame  : lambda x: my_len(x.values),
                    pd.core.groupby.generic.DataFrameGroupBy : lambda x: x.ngroups
                    }

def get_data(x):
    type_debug = type(x)
    method = get_data_switchDict.get(type_debug,lambda x: np.asarray(x,dtype='float'))
    return method(x)

def get_float_nan(x, **k):
    if isinstance(x,list):return [get_float_nan(y,**k) for y in x]
    if isinstance(x,np.ndarray):return [get_float_nan(y,**k) for y in x]
    out = float(x)
    # out = np.array(x, dtype=float)
    nan_val = k.get("nan_val",np.nan)
    if pd.isnull(out) : 
        out = nan_val
    return out

def get_float(x, **kwargs):
    type_ = type(x)
    method = get_float_switchDict.get(type_,lambda x, **k : get_float_nan(x,**k))
    out = get_float_nan(method(x,**kwargs),**kwargs)
    return out

def get_date(x,**kwargs):
    if isinstance(x,list): return [get_date(n,**kwargs) for n in x]
    type_debug = type(x)
    method = get_date_switchDict.get(type_debug,None)
    return method(x,**kwargs)

def my_len(x):
    if is_primitive(x):return 0
    type_ = type(x)
    method = my_len_switchDict.get(type_,lambda x : 0)
    return method(x)

primitive = (int, str, bool,np.float32,np.float64,float)
def is_primitive(thing):
    debug = type(thing)
    return isinstance(thing, primitive)

def get_matrix(x):
    if x is None: return []
    if isinstance(x,list): return [get_matrix(y) for y in x]
    # if isinstance(x,list): return np.array([np.array(y) for y in x])
    if isinstance(x,tuple): return [get_matrix(y) for y in x]
    x = get_data(x)
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