# from include_all import *

import pandas as pd
import numpy as np
from metrics import *
from selection import column_selector, select_constant_columns
from core import *
from data_conversion import get_data, my_len


def get_bound_box(mat: np.ndarray,coeff = None) -> np.ndarray:
    """
    Computes the bounding box for a given matrix.

    This function calculates the minimum and maximum values along each dimension (column) of the input matrix.
    Optionally, it can scale the bounding box by a specified coefficient around the mean value.

    Args:
        mat (np.ndarray): A 2D NumPy array for which the bounding box is to be computed.
        coeff (float, optional): A scaling coefficient for the bounding box. If provided, the bounding box is
                                scaled around its mean value. Default is None.

    Returns:
        np.ndarray: A 2D NumPy array where the first row contains the minimum values and the second row contains
                    the maximum values for each dimension (column) of the input matrix.

    Example:
        
        >>> mat = np.array([[1, 2], [3, 4], [5, 6]])
        >>> bounding_box = get_bound_box(mat)
        # bounding_box will be array([[1, 2], [5, 6]])
    """
    out = np.array([(min(mat[:,d]), max(mat[:,d])) for d in range(mat.shape[1]) ]).T
    if coeff is not None:
        mean = np.mean(out,axis=0)
        out = (out -mean)*coeff + mean
    return out

def bound_box(mat: np.ndarray,bounding_box: np.ndarray) -> np.ndarray:
    """
    Applies a bounding box to each point in a given matrix.

    This function modifies the input matrix such that each element is constrained within the limits defined by
    the provided bounding box. Each dimension (column) of the matrix is bounded individually.

    Args:
        mat (np.ndarray): A 2D NumPy array to which the bounding box constraints are to be applied.
        bounding_box (np.ndarray): A 2D NumPy array representing the bounding box, where the first row contains 
                                the minimum values and the second row contains the maximum values for each dimension.

    Returns:
        np.ndarray: The modified matrix with each element constrained within the bounding box.

    Example:
        
        >>> mat = np.array([[0, 7], [2, 5], [6, 1]])
        >>> bounding_box = np.array([[1, 2], [5, 6]])
        >>> constrained_mat = bound_box(mat, bounding_box)
        
        constrained_mat will be 
        array([[1, 6], [2, 5], [5, 2]])
    """
    out = mat.copy()
    def helper(n,d):
        out[n,d] = min(max(out[n,d],bounding_box[0,d]),bounding_box[1,d])
    [helper(n,d) for n in range(mat.shape[0]) for d in range(mat.shape[1]) ]
    return out

def get_matching_cols(a,match):
    if  isinstance(match,list): 
        out= []
        def fun(out,m):out+=m
        [fun(out,get_matching_cols(a,m)) for m in match]
        return out
    return [col for col in a.columns if match in col]

def variable_selector(**params) -> dict:
    """
    Implements a greedy algorithm for feature selection in a dataset, aiming to minimize the prediction error.

    This function selects the most relevant variables (features) from a dataset based on a greedy approach. 
    It iteratively evaluates the impact of adding each variable on the prediction error, retaining those 
    that contribute to its minimization. This method looks for local minima.

    Args:
        **params: Keyword arguments including:
            x, y, z (pd.DataFrame): DataFrames representing the input data.
            fx, fz (np.ndarray): Arrays representing function values associated with x and z.
            error_fun (callable, optional): The function used to compute the error. Default is `get_mean_squared_error`.
            predictor (callable, optional): The prediction function whose error is to be minimized. Default is `op.projection`.
            variables_selector_csv (str, optional): Path to save a CSV file with selected variables and errors. Default is an empty string.
            selector_cols (list, optional): List of column names to be used for selection. Default is an empty list.
            cols_drop (list, optional): List of column names to be excluded from the selection. Default is an empty list.
            keep_columns (list, optional): List of column names that must always be kept. Default is an empty list.

    Returns:
        dict: A list of column names that are selected as the most relevant for minimizing the prediction error.

    Example:
    
        Example usage

        >>> selected_cols = variable_selector(x=df_x, y=df_y, z=df_z, fx=array_fx, fz=array_fz, predictor=my_predictor_function)
    """
    kwargs = params.copy()
    x,y,z,fx,fz = kwargs['x'],kwargs['y'],kwargs['z'],kwargs['fx'],kwargs['fz']
    error_fun,predictor = kwargs.get('error_fun',get_mean_squared_error),kwargs.get('predictor',op.projection)
    variable_selector_csv = kwargs.get('variables_selector_csv',[])
    matching_cols = kwargs.get('selector_cols',[])

    cols_drop = params.get('cols_drop',[])
    cols_drop = set(cols_drop) | set(select_constant_columns(x))
    params['cols_drop'] = list(cols_drop)

    x,y,z = column_selector([x,y,z],**params)
    
    xyzcolumns = list(x.columns)
    keep_columns = kwargs.get("keep_columns",[])
    keep_columns = get_matching_cols(x,keep_columns)
    matching_cols = get_matching_cols(x,matching_cols)

    def helper(x,y,z,fx,fz,cols):
        kwargs['cols_keep'] = cols
        try:
            f_z = predictor(**kwargs)
            erreur = error_fun(fz, f_z)
        except Exception as e:
            print(e)
            erreur = float('Inf')
        print("columns:",cols, " -- erreur:",erreur)
        return erreur

    cols = list(set(x.columns).difference(set(keep_columns+matching_cols)))
    def fun(col):
        # values = set(x[col])
        erreur = helper(x,y,z,fx,fz,col)
        # if len(values) > 1:
        #     return helper(x,y,z,fx,fz,col)
        # else:
        #     erreur = float('Inf')
        #     print("columns:",col, " -- erreur:",erreur)
        return erreur
    erreurs = np.asarray([fun(keep_columns + [col]) for col in cols])
    erreurs, order = lexicographical_permutation(erreurs)
    best_erreur,best_erreurs = erreurs[0],[erreurs[0]]*(len(keep_columns)+1)
    best_col, best_cols = cols[order[0]],keep_columns + [cols[order[0]]]
    order = list(order)+list(range(len(cols),len(cols)+len(matching_cols)))
    cols += matching_cols
    for n in range(0,len(cols)):
        col = cols[order[n]]
        if col not in best_cols :
            best_cols.append(col)
            erreur = helper(x,y,z,fx,fz,best_cols) 
            if erreur >= best_erreur: best_cols.remove(col)
            else : 
                best_erreur = erreur
                best_erreurs.append(erreur)
    output = { 'keep_columns' : best_cols, 'errors' : best_erreurs}
    
    if len(variable_selector_csv): 
        # csv_file = output.copy()
        # csv_file['keep_columns'].insert(0,"NoNe") 
        # csv_file['errors'].insert(0,erreurs) 
        pd.DataFrame(output).to_csv(variable_selector_csv,sep = ',')
    return output['keep_columns']

def lexicographical_permutation(x,fx=[],**kwargs) -> tuple:
    """
    Sorts the given array `x` (and optionally, the associated `fx` array) in lexicographical order.

    This function sorts `x` based on its values (if 1D) or on the values of one of its columns (if 2D). If an
    `fx` array is provided, it is sorted in the same order as `x`. The function returns the sorted arrays along
    with an index array that describes the new ordering.

    Args:
        x (np.ndarray): A one-dimensional or two-dimensional NumPy array to be sorted.
        fx (np.ndarray or list, optional): An array or list with the same length as `x` that will be reordered 
            in the same way as `x`. Default is an empty list.
        **kwargs: Additional keyword arguments. Can include:
            - indexfx (int): The column index of `x` to be used for sorting if `x` is a 2D array. 
                            Default is 0 (the first column).

    Returns:
        tuple: A tuple containing the sorted `x` array, optionally the reordered `fx`, and the index array 
            that represents the new order. If `fx` is not provided or its length does not match `x`, 
            the function returns only the sorted `x` and the index array.

    Example:
    
        Sorting a 1D array
        
        >>> x = np.array([3, 1, 2])
        >>> x_sorted, index_array = lexicographical_permutation(x)

        Sorting a 2D array along with a 1D array `fx`
        
        >>> x_2d = np.array([[3, 'a'], [1, 'b'], [2, 'c']])
        >>> fx = np.array([30, 10, 20])
        >>> x_sorted, fx_sorted, index_array = lexicographical_permutation(x_2d, fx, indexfx=0)
    """
    # x = get_data(x)
    if x.ndim==1: index_array = np.argsort(x)
    else: 
        indexfx = kwargs.get("indexfx",0)
        index_array = np.argsort(a = x[:,indexfx])
    x_sorted = x[index_array]
    if (my_len(fx) != my_len(x_sorted)): return (x_sorted,index_array)
    if type(fx) == type([]): out = [s[index_array] for s in fx]
    # else: out = np.take_along_axis(arr = x,indices = index_array,axis = indexx)
    else: out = fx[index_array]
    return (x_sorted,out,index_array)

def hot_encoder(data_frame : pd.DataFrame,cat_cols_include = [],cat_cols_exclude = []) -> pd.DataFrame:
    # data_frame.to_csv (r'data_frame.csv', header=True)
    num_dataframe = data_frame.select_dtypes(include='number')
    num_cols = set(num_dataframe.columns)
    if len(cat_cols_include):num_cols.difference_update(cat_cols_include)
    cat_cols = set(data_frame.columns)
    cat_cols = cat_cols.difference(num_cols)
    if len(cat_cols_exclude): cat_cols = cat_cols.difference(cat_cols_exclude)
    cat_dataframe = data_frame[list(cat_cols)]
    num_dataframe = data_frame[list(num_cols)]
    index = cat_dataframe.index
    values =cat_dataframe.to_numpy(dtype = str)
    cols = np.array(cat_dataframe.columns,dtype=str)
    (cat_dataframe,cat_columns) = cd.tools.hot_encoder(values,cols)
    cat_dataframe = pd.DataFrame(cat_dataframe, columns = cat_columns, index = index)
    if len(num_cols) :
        if not cat_dataframe.empty :
            cat_dataframe = pd.concat([num_dataframe,cat_dataframe], axis=1,join="inner")
        else :
            cat_dataframe = num_dataframe
    # cat_dataframe.to_csv (r'hot_encoder.csv', header=True)
    return cat_dataframe

def unity_partition(fx: np.ndarray, unique_values = []) -> np.ndarray:
    """
    Creates a unity partition (one-hot encoding) of the input array `fx` based on its unique values.

    This function transforms the input array `fx` into a matrix where each column corresponds to 
    one unique value from `fx`. Each row in the output matrix represents an element in `fx`, 
    with a '1' in the column corresponding to its value and '0's elsewhere.

    Args:
        fx (np.ndarray): A NumPy array containing the values to be partitioned.
        unique_values (list, optional): A list of unique values to be used for partitioning. 
                                        If not provided, the unique values will be determined from `fx`.
                                        Default is an empty list.

    Returns:
        np.ndarray: A 2D NumPy array where each row corresponds to an element in `fx` and each column 
                    corresponds to a unique value in `fx`. The entries in the matrix are '1' where the 
                    row's original value matches the column's unique value, and '0' otherwise.

    Example:
        
        >>> fx = np.array([1, 2, 1, 3])
        >>> partitioned_fx = unity_partition(fx)
        
        This will return 
        array([[1., 0., 0.],
                [0., 1., 0.],
                [1., 0., 0.],
                [0., 0., 1.]])

    Note:
    This function is typically used for categorical data encoding in machine learning and data analysis tasks.
    """
    if len(unique_values) == 0: unique_values= np.unique(get_data(fx).flatten())
    P = len(unique_values)
    N = len(fx)
    out = np.zeros((N,P))
    d=0
    for v in unique_values :
        indices = np.where(fx == v)
        out[indices[0],d]=1.
        d=d+1
    return out