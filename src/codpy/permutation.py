import warnings
from typing import List

import numpy as np
import pandas as pd
import xarray
from codpydll import *
from scipy.optimize import linear_sum_assignment

from codpy.core import _requires_rescale, op
from codpy.data_conversion import get_matrix
from codpy.dictionary import Dict, Set, cast
from codpy.random_utils import random_select_interface
from codpy.sampling import sharp_discrepancy
from codpy.selection import column_selector
from codpy.utils import format_23, format_32


def map_invertion(map, type_in=None):
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
    if type_in is None:
        type_in = type(map)
    out = cast(data=map, type_in=type_in, type_out=Dict[int, Set[int]])
    out = cd.alg.map_invertion(out)
    return out
    # return cast(out, type_out=type_in, type_in=Dict[int, Set[int]])


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
    D = np.min((N, C.shape[1]))
    permutation = linear_sum_assignment(C, maximize=False)
    out = np.array(range(0, D))
    for n in range(0, D):
        out[permutation[1][n]] = permutation[0][n]
    return out


def lsap(C: np.ndarray, sub=False) -> np.ndarray:
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
    return np.array(cd.alg.LSAP(C, sub))


def grid_projection(**kwargs):
    x = kwargs.get("x", [])
    grid_projection_switchDict = {
        pd.DataFrame: lambda x, **kwargs: grid_projection_dataframe(x),
        xarray.core.dataarray.DataArray: lambda x, **kwargs: grid_projection_xarray(x),
    }

    def grid_projection_dataframe(**kwargs):
        x = kwargs.get("x", [])
        out = cd.alg.grid_projection(x.values)
        out = pd.DataFrame(out, columns=x.columns, index=x.index)
        return out

    def grid_projection_xarray(**kwargs):
        x = kwargs.get("x", [])
        index_string = kwargs.get("index", "N")
        indexes = x[index_string]
        out = x.copy()
        for index in indexes:
            mat = x[index_string == int(index)].values
            mat = cd.alg.grid_projection(mat.T)
            out[index_string == int(index)] = mat.T
        return out

    def grid_projection_np(**kwargs):
        x = kwargs.get("x", [])
        if x.ndim == 3:
            shapes = x.shape
            x = format_32(x)
            out = cd.alg.grid_projection(x)
            out = format_23(out, shapes)
            return out
        else:
            return cd.alg.grid_projection(get_matrix(x))

    type_debug = type(x)
    method = grid_projection_switchDict.get(
        type_debug, lambda x: grid_projection_np(**kwargs)
    )
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
            x = random_select_interface(
                x, xmax=sharp_discrepancy_xmax, seed=sharp_discrepancy_seed
            )
        kernel.init(x=x)
        out = cd.alg.match(get_matrix(x), Ny)
        return out

    if isinstance(x, pd.DataFrame):
        return match_dataframe(x)
    elif isinstance(x, np.ndarray):
        return match_array(x)
    else:
        raise ValueError(
            "Invalid type for x. Only numpy arrays and pandas DataFrames are supported."
        )


if __name__ == "__main__":
    import time

    import core
    import pandas as pd

    N, D = [2**n for n in range(8, 15)], 3
    times = []
    for n in N:
        A = np.random.normal(size=[n, D])
        B = np.random.normal(size=[n, D])
        C = core.op.Dnm(A, B, distance="norm22")
        start = time.time()
        permutation = lsap(C, False)
        end = time.time()
        times.append(end - start)
    result = pd.DataFrame({"size": N, "times": times})
    print(result)
    pass
