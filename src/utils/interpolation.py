from scipy import interpolate as sci_interpolate
from codpy.utils.data_conversion import get_float

def my_range(start, stop, step):
    start_type = type(start)
    out=[]
    while start < stop:
        out.append(start)
        start += step
    return out
def interpolate(x, fx, z, **kwargs):
    x, fx, z = get_float(x), get_float(fx), get_float(z)
    if len(x) == 1: x.append(x[0]+1.), fx.append(fx[0])
    return sci_interpolate.interp1d(x, fx, **kwargs)(z) 

def interpolate1D(x, fx, z, **kwargs):
    kind = str(kwargs.get("kind","linear"))
    bounds_error = bool(kwargs.get('bounds_error',False))
    copy = bool(kwargs.get('copy',False))
    var_col = kwargs.get('var_col',None)
    float_fun = kwargs.get('float_fun',None)
    fz = pd.DataFrame(columns = fx.columns)

    cols = fz.columns
    for col in cols:
        fz[col] = interpolate(x, fx[col], z,kind = kind, bounds_error = bounds_error, fill_value= (fx[col][0],fx[col][-1]), copy=copy)
        pass 
    return fz

def interpolate_nulls(data,**kwargs):
    kind = str(kwargs.get("kind","linear"))
    bounds_error = bool(kwargs.get('bounds_error',False))
    copy = bool(kwargs.get('copy',False))
    var_col = kwargs.get('var_col',None)
    float_fun = kwargs.get('float_fun',None)

    nulls = [col for col in data.columns if data[col].isnull().sum()]
    for col in nulls:
        fx = data.loc[data[col].notnull()][col].values
        if var_col is None:
            x = data.loc[data[col].notnull()].index.values
            z = data.index.values
        else: 
            x = data.loc[data[col].notnull()][var_col].values
            z = data[var_col].values
        if float_fun is not None: x,z=float_fun(x),float_fun(z)
        data[col] = interpolate(x, fx, z,kind = kind, bounds_error = bounds_error, fill_value= (fx[0],fx[-1]), copy=copy)
        pass 
    return data