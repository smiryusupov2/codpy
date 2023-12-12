import pandas as pd
import numpy as np
import codpydll
import codpypyd as cd
from codpy.src.core import kernel, op
import xarray
from scipy.optimize import linear_sum_assignment
from codpy.utils.data_conversion import get_matrix
from codpy.utils.selection import column_selector
from codpy.utils.utils import format_32, format_23
from codpy.src.sampling import sharp_discrepancy
from codpy.utils.dictionary import cast, Dict, Set

def map_invertion(map,type_in = None):
    """
    Invert a mapping, transforming a map from one distribution to another into its inverse.

    Args:
        map (dict or similar): The mapping to invert.
        type_in (type, optional): The type of the input map. If None, the type of 'map' is used.

    Returns:
        The inverted map, with the type specified by 'type_in'.

    Example:

        >>> original_map = {...}
        >>> inverted_map = map_inversion(original_map)
    """
    if type_in is None: type_in = type(map)
    out = cast(data = map, type_in = type_in, type_out = Dict[int, Set[int]])
    out = cd.alg.map_invertion(out)
    return cast(out,type_out = type_in, type_in = Dict[int, Set[int]])

def reordering(**kwargs):
    """
    Reorder elements in one distribution (source) to align with another distribution (target).
    This function can handle both NumPy arrays and pandas DataFrames. It uses different strategies
    for reordering based on the type of the input data.

    For DataFrames, it internally calls `reordering_dataframe`. For NumPy arrays, it calls `reordering_np`.

    Args:
        **kwargs: Arbitrary keyword arguments.
            x (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Source distribution.
            y (:class:`numpy.ndarray` or :class:`pandas.DataFrame`): Target distribution.
            permut (str, optional): Specifies the permutation strategy ('source' or other). Default is 'source'.
            iter (int, optional): Number of iterations for the encoder, applicable for NumPy arrays. Default is 10.
            lsap_fun (function, optional): Function to compute the Linear Sum Assignment Problem (LSAP).

    Returns:
        Tuple: A tuple (x_, y_, permutation), where x_ and y_ are reordered versions of the source and target,
            and permutation is the mapping from source to target.

    Example:
        Example with DataFrames
        
        >>> source_df = pd.DataFrame([...])
        >>> target_df = pd.DataFrame([...])
        >>> reordered_source, reordered_target, permutation = reordering(x=source_df, y=target_df)

        Example with NumPy arrays
        
        >>> source_array = np.array([...])
        >>> target_array = np.array([...])
        >>> reordered_source, reordered_target, permutation = reordering(x=source_array, y=target_array)
    """
    reordering_format_switchDict = { pd.DataFrame: lambda **kwargs :  reordering_dataframe(**kwargs) }
    def reordering_dataframe(**kwargs):
        x,y = kwargs.get('x',[]),kwargs.get('y',[])
        a,b,permutation = reordering(**{**kwargs,**{'x':x.values,'y':y.values}})
        x,y = pd.DataFrame(a,columns = x.columns, index = x.index),pd.DataFrame(b,columns = y.columns, index = y.index)
        return x,y,permutation
    def reordering_np(**kwargs):
        x,y = kwargs.get('x',[]),kwargs.get('y',[])
        permut=kwargs.get('permut','source')

        if x.shape[1] != y.shape[1]:
            kernel.init(**{**kwargs,**{'x':x,'y':None,'z':None}})
            permutation = cd.alg.encoder(get_matrix(x),get_matrix(y),kwargs.get('iter',10))
            if permut != 'source':x_,y_ = x,y[permutation]
            else: 
                permutation = map_invertion(permutation, type_in = np.ndarray)
                x_,y_ = x[permutation],y
        else:
            kernel.init(**{**kwargs,**{'x':x,'y':y,'z':None}})
            D = op.Dnm(**{**kwargs,**{'x':x,'y':y}})
            # test = D.trace().sum()
            lsap_fun = kwargs.get("lsap_fun", lsap)
            permutation = lsap_fun(D)
            if permut != 'source':
                permutation = map_invertion(permutation, type_in = np.ndarray)
                x_,y_ = x,y[permutation]
            else: x_,y_ = x[permutation],y
        # D = op.Dnm(x,y,**kwargs)
        # test = D.trace().sum()
        return x_,y_,permutation
    type_debug = type(kwargs['x'])
    method = reordering_format_switchDict.get(type_debug, reordering_np)
    return method(**kwargs)

def grid_projection(**kwargs):
    x = kwargs.get('x',[])
    grid_projection_switchDict = { pd.DataFrame: lambda x,**kwargs :  grid_projection_dataframe(x),
                                    xarray.core.dataarray.DataArray: lambda x,**kwargs :  grid_projection_xarray(x) }
    def grid_projection_dataframe(**kwargs):
        x = kwargs.get('x',[])
        out = cd.alg.grid_projection(x.values)
        out = pd.DataFrame(out,columns = x.columns, index = x.index)
        return out
    def grid_projection_xarray(**kwargs):
        x = kwargs.get('x',[])
        index_string = kwargs.get("index","N")
        indexes = x[index_string]
        out = x.copy()
        for index in indexes:
            mat = x[index_string==int(index)].values
            mat = cd.alg.grid_projection(mat.T)
            out[index_string==int(index)] = mat.T
        return out
    def grid_projection_np(**kwargs):
        x = kwargs.get('x',[])
        if x.ndim==3:
            shapes = x.shape
            x = format_32(x)
            out = cd.alg.grid_projection(x)
            out = format_23(out,shapes)
            return out
        else: return cd.alg.grid_projection(get_matrix(x))
    type_debug = type(x)
    method = grid_projection_switchDict.get(type_debug,lambda x : grid_projection_np(**kwargs))
    return method(x)

def scipy_lsap(C):
    """
    Solve the Linear Sum Assignment Problem (LSAP) using the SciPy optimization module.

    This function finds an optimal assignment that minimizes the total cost based on the
    cost matrix C. It uses the `linear_sum_assignment` method from SciPy, which implements
    the Hungarian algorithm (or Munkres algorithm) for this purpose.

    Args:
        C (:class:`numpy.ndarray`): A 2D array representing the cost matrix. Each element C[i, j] is the cost 
                    of assigning the ith worker to the jth job.

    Returns:
        :class:`numpy.ndarray`: An array representing the optimal assignment. For each job (column in the cost
                matrix), it gives the index of the worker assigned to that job.

    Example:
        Create a cost matrix
        
        >>> cost_matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])

        Solve the LSAP
        
        >>> optimal_assignment = scipy_lsap(cost_matrix)
        >>> print(optimal_assignment)  
        # Output: [1, 0, 2]
    """
    N = C.shape[0]
    D = np.min((N,C.shape[1]))
    permutation = linear_sum_assignment(C, maximize=False)
    out = np.array(range(0,D))
    for n in range(0,D):
        out[permutation[1][n]] = permutation[0][n]
    return out

def lsap(x):
    return cd.alg.LSAP(x)

def get_rand(kwargs):
    grid_projection = kwargs.get("grid_projection", False)
    seed = kwargs.get("seed",None)
    rand = kwargs.get("random_sample",np.random.random_sample)
    N = kwargs.get("Nx",None)
    D = kwargs.get("Dx",None)
    if N is None or D is None:
        x = column_selector(kwargs['x'],**kwargs)
        N = kwargs.get("Nx",x.shape[0])
        D = kwargs.get("Dx",x.shape[1])
    if seed: np.random.seed(seed)
    shape = [N,D]
    x = rand(shape)
    if grid_projection: x = grid_projection(x=x)
    return x

def _get_x(fx,**kwargs):
    x = kwargs.get('get_rand', get_rand)({**kwargs,**{'x':fx}})
    Dx = kwargs.get("Dx",fx.shape[1])
    Dfx = fx.shape[1]
    if (kwargs.get("reordering",True)):
        kwargs['x'],kwargs['y'] = x,fx
        if Dx==Dfx:
            kwargs['distance'] = "norm22"
            x,y,permutation = reordering(**kwargs, permut = 'source')
        else:
            def helper(**kwargs): return gen.encoder(**kwargs).params['fx']
            x = kwargs.get("encoder",helper)(**kwargs,permut='source')
    return x

def _get_y(x,**kwargs):
    Ny = kwargs.get("Ny",x.shape[0])
    if Ny < x.shape[0]: return kwargs.get("cluster_fun", sharp_discrepancy)(x,**kwargs)
    return x

def _get_z(x,**kwargs):
    rand = kwargs.get("random_sample",np.random.random_sample)
    Nz = kwargs.get("Nz",x.shape[0])
    Dz = kwargs.get("Dx",x.shape[1])
    shape = [Nz,Dz]
    z = rand(shape)
    return z

def _get_xyz(fx,**kwargs):
    kwargs['fx'] = fx
    kwargs['x'] = kwargs.get("get_x", _get_x)(**kwargs)
    kwargs['y'] = kwargs.get("get_y", _get_y)(**kwargs)
    z = kwargs.get("get_z", _get_z)(**kwargs)

    return kwargs['x'], kwargs['y'], z