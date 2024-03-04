import pandas as pd
import numpy as np
import warnings
import xarray
from scipy.optimize import linear_sum_assignment
from typing import List  
from core import kernel, op, kernel_helper2, _requires_rescale
from sampling import sharp_discrepancy
from data_conversion import get_matrix
from selection import column_selector
from utils import format_32, format_23
from dictionary import cast, Dict, Set
from random_utils import random_select_interface
  

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

def scipy_lsap(C: np.ndarray) -> np.ndarray:
    """
    Solve the Linear Sum Assignment Problem (LSAP) using the SciPy optimization module.

    This function finds an optimal assignment that minimizes the total cost based on the
    cost matrix C. It uses the `linear_sum_assignment` method from SciPy, which implements
    the Hungarian algorithm (or Munkres algorithm) for this purpose.

    Args:
        C (:class:`numpy.ndarray`): A 2D array representing the cost matrix. Each element ``C[i, j]`` is the cost 
                    of assigning the ith worker to the jth job.

    Returns:
        :class:`numpy.ndarray`: An array representing the optimal assignment. For each job (column in the cost matrix), it gives the index of the worker assigned to that job.

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


def lsap(C: np.ndarray) -> np.ndarray:
    """
    Solve the Linear Sum Assignment Problem (LSAP) using CodPy's optimization module.

    This function finds an optimal assignment that minimizes the total cost based on the
    cost matrix C. It uses the `linear_sum_assignment` method from CodPy, which implements
    the Hungarian algorithm (or Munkres algorithm) for this purpose.

    Args:
        C (:class:`numpy.ndarray`): A 2D array representing the cost matrix. Each element ``C[i, j]`` is the cost 
                    of assigning the ith worker to the jth job.

    Returns:
        :class:`numpy.ndarray`: An array representing the optimal assignment. For each job (column in the cost matrix), it gives the index of the worker assigned to that job.

    Example:
        Create a cost matrix
        
        >>> cost_matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])

        Solve the LSAP
        
        >>> optimal_assignment = scipy_lsap(cost_matrix)
        >>> print(optimal_assignment)  
        # Output: [1, 0, 2]
    """
    return np.array(cd.alg.LSAP(C))

def reordering(x: np.ndarray, y: np.ndarray, permut: str ='source', iter=10, lsap_fun = None):
    """
    Reorder elements in one distribution (source) to align with another distribution (target).
    This function can handle both NumPy arrays and pandas DataFrames.
    
    Args:
        x (numpy.ndarray or pandas.DataFrame): Source distribution.
        y (numpy.ndarray or pandas.DataFrame): Target distribution.
        permut (str): Specifies the permutation strategy ('source' or other).
        iter (int): Number of iterations for the encoder, applicable for NumPy arrays.
        lsap_fun (callable): Function to compute the Linear Sum Assignment Problem (LSAP).

    Returns:
        Tuple containing reordered x, reordered y, and the permutation mapping.
    """
    if lsap_fun is None:
        lsap_fun = lsap
    def reordering_dataframe(x: pd.DataFrame = [], y: pd.DataFrame = [], permut = 'source', iter = 10, lsap_fun=None):
        a,b,permutation = reordering(x.values, y.values, permute = permut, iter=iter, lsap_fun=lsap_fun)
        x,y = pd.DataFrame(a,columns = x.columns, index = x.index),pd.DataFrame(b,columns = y.columns, index = y.index)
        return x,y,permutation

    def reordering_np(x: np.ndarray, y: np.ndarray, permut, iter = 10, lsap_fun = lsap):
        if x.shape[1] != y.shape[1]:
            kernel.init(x = x,y =None, z = None)
            permutation = cd.alg.encoder(get_matrix(x),get_matrix(y), iter = iter)
            if permut != 'source':x_,y_ = x,y[permutation]
            else: 
                permutation = map_invertion(permutation, type_in = np.ndarray)
                x_,y_ = x[permutation],y
        else:
            kernel.init(x = x,y =y, z = None)
            D = op.Dnm(x = x, y = y)
            permutation = lsap_fun(D)
            if permut != 'source':
                permutation = map_invertion(permutation, type_in = np.ndarray)
                x_,y_ = x,y[permutation]
            else: x_,y_ = x[permutation],y
        # D = op.Dnm(x,y,**kwargs)
        # test = D.trace().sum()
        return x_,y_,permutation

    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        return reordering_dataframe(x, y)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return reordering_np(x, y, permut=permut, iter = iter, lsap_fun=lsap_fun)
    else:
        raise ValueError("Invalid types for x and y. Only numpy arrays and pandas DataFrames are supported.")

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


def match(x, Ny=None, sharp_discrepancy_xmax=None, sharp_discrepancy_seed=None):
    """
    Match function to resample or reorder data points in x to match a specified distribution size Ny.
    
    Args:
        x (numpy.ndarray or pandas.DataFrame): Input data points to be matched.
        Ny (int): Number of points to match to, if None, matches to the size of ``x``.
        sharp_discrepancy_xmax (int): Optional parameter for sharp discrepancy, max value.
        sharp_discrepancy_seed (int): Optional seed value for random selection in sharp discrepancy.

    Returns:
        numpy.ndarray or pandas.DataFrame: Matched data points.
    """
    x = column_selector(x)
    if Ny is None:
        Ny = x.shape[0]
    if Ny >= x.shape[0]: 
        return x

    def match_dataframe(x):
        out = match(get_matrix(x))
        return pd.DataFrame(out, columns=x.columns)

    def match_array(x):
        if sharp_discrepancy_xmax is not None:
            x = random_select_interface(x, xmax=sharp_discrepancy_xmax, seed=sharp_discrepancy_seed)
        kernel.init(x=x) 
        out = cd.alg.match(get_matrix(x), Ny)
        return out

    if isinstance(x, pd.DataFrame):
        return match_dataframe(x)
    elif isinstance(x, np.ndarray):
        return match_array(x)
    else:
        raise ValueError("Invalid type for x. Only numpy arrays and pandas DataFrames are supported.")

class encoder:
    """
    A class for encoding data into a lower-dimensional subspace using kernel methods.

    The encoder class is designed to transform the input data ``x`` into a lower-dimensional representation ``y``
    by minimizing the mean-square curvature, effectively smoothing and reducing the dimensionality of the data.
    It outputs a permutation that minimizes the norm of the gradient of the function, serving as an optimal
    encoding in the reduced space.

    Args:
        y (array_like): The data to be encoded.
        x (array_like, optional): Additional data points to be considered in encoding. Defaults to None, 
                                  in which case ``x`` is set to ``y``.
        Dx (int, optional): The desired dimensionality of the encoded subspace. If specified and different
                            from the dimensionality of ``x``, ``x`` will be matched to this dimension. Defaults to None.
        permut (str, optional): Specifies the permutation strategy, either 'source' or another specified method. 
                                Defaults to ``'source'``.
        kernel_fun (str, optional): The kernel function to use for encoding. Defaults to ``'tensornorm'``.
        map (str, optional): The mapping function to use with the kernel. Defaults to ``'unitcube'``.
        polynomial_order (int, optional): The polynomial order for the kernel function. Defaults to ``2``.
        regularization (float, optional): The regularization parameter for the kernel. Defaults to ``1e-8``.
        rescale (bool, optional): Whether to rescale the data. Defaults to ``False``.
        rescale_params (dict, optional): Parameters for the rescaling process if rescale is True. 
                                         Defaults to ``{'max': 1000, 'seed': 42}``.

    Usage:

        >>> x = np.random.randn(100, 1)
        >>> z = np.random.randn(100, 1)
        >>> enc = encoder(y=x, x=z)
        >>> encoded_data = enc(z)  # Project new data into the encoded space

    Note:

        The encoder uses kernel methods to perform a non-linear dimensionality reduction, particularly useful when
        the features in the dataset exhibit non-linear relationships. The choice of kernel and mapping function
        can significantly affect the encoding quality and should be selected based on the nature of the data and 
        the specific requirements of the application.
    """
    def __init__(self,y: np.ndarray,x: np.ndarray = None, Dx: int = None, permut: str='source', 
                 kernel_fun: str = "tensornorm", map: str = "unitcube", polynomial_order:int = 2, 
                 regularization:float = 1e-8, rescale:bool = False, rescale_params: dict = {'max': 1000, 'seed':42},
                 verbose = False):
        self.kernel_fun, self.map = kernel_fun, map
        y,x = column_selector(y),column_selector(x)
        if x is None:
            x = y
            if Dx is not None and Dx != x.shape[1]:
                x = match(y.T,Ny=Dx).T
            else:return

        params = {'rescalekernel':{'max': 1000, 'seed':42},
        'set_codpykernel' : kernel_helper2(kernel=kernel_fun, map= map, polynomial_order=polynomial_order, regularization=regularization),
        'rescale': rescale,
        }
        if rescale == True or _requires_rescale(map_name=map):
            params['rescale'] = True
            params['rescalekernel'] = rescale_params
            if verbose:
                warnings.warn("Rescaling is set to True as it is required for the chosen map.")
            kernel.init(x,y,x, **params)
        else:
            params['rescale'] = rescale
            kernel.init(**params)

        permutation = cd.alg.encoder(get_matrix(x),get_matrix(y))
        self.permutation = permutation
        if permut == 'source': 
            permut = map_invertion(permutation,type_in = List[int])
            if isinstance(x,pd.DataFrame): x= pd.DataFrame(x.iloc[permut], columns = x.columns)
            else: x=x[permut]
            self.x, self.y, self.fx = y, y, x
        else:
            if isinstance(y,pd.DataFrame): y  = pd.DataFrame(y.iloc[permutation], columns = y.columns)
            else: y=y[permutation]
            self.x, self.y, self.fx =  y, y, x

    def __call__(self, z):
        return op.projection(self.x, self.y, z, self.fx, kernel_fun=self.kernel_fun, map=self.map)

class decoder:
    """
    A class for decoding data from a lower-dimensional subspace back to the original space using kernel methods.

    The decoder transforms the encoded data back into the original high-dimensional space. It essentially learns and applies
    the inverse mapping of the encoding process.

    Args:
        x (array_like): The encoded input data.
        y (array_like): The encoded data, acting here as a base for the reconstruction.
        fx (array_like): The original input data to be reconstructed.

    Methods:
        __call__(z, kernel_fun, map): Performs projection of new encoded data ``z`` back into the original space using the specified kernel and map.

    Usage:
        >>> x = np.random.randn(100, 1)
        >>> z = np.random.randn(100, 1)
        >>> enc = encoder(x, z)
        >>> reconstructed_data = dec(z)
    """
    def __init__(self,encoder: object):
        self.x, self.y, self.fx= encoder.fx ,encoder.fx, encoder.x

    def __call__(self, z: np.ndarray, kernel_fun: str = "tensornorm", map: str = "unitcube"):
        """
        Apply the decoding to encoded data points.

        Args:
            z (array_like): The encoded data to project back into original space.
            kernel_fun (str): Specifies the kernel function to use for projection.
            map (str): Specifies the mapping function to use for projection.

        Returns:
            array_like: The projected data in the original space.
        """
        return op.projection(x = self.x, y = self.y, z = z, fx = self.fx, kernel_fun=kernel_fun,
                                map=map)



def get_rand(x, grid_projection = False, seed = None, rand = np.random.random_sample, **kwargs):
    N,D = x.shape
    if N is None or D is None:
        x = column_selector(x,**kwargs)
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



